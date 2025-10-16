import torch
from torchvision import datasets, transforms
import open_clip
from tqdm import tqdm
import pandas as pd

# ===============================
# 参数设置
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型列表
MODEL_LIST = [
    ["RN50", "openai"],
    ["RN50", "cc12m"],
    ["RN101", "openai"],
    ["RN101", "yfcc15m"],
    ["RN101-quickgelu", "openai"],
    ["RN101-quickgelu", "yfcc15m"],
    ["RN50x4", "openai"],
    ["RN50x64", "openai"],
    ["ViT-B-32", "openai"],
    ["ViT-B-32", "laion2b_e16"],
    ["ViT-B-32", "datacomp_xl_s13b_b90k"],
    ["ViT-B-32", "commonpool_m_clip_s128m_b4k"],
    ["ViT-B-32-256", "datacomp_s34b_b86k"],
    ["ViT-B-32-quickgelu", "laion400m_e31"],
    ["ViT-B-32-quickgelu", "metaclip_fullcc"],
    ["ViT-B-16", "openai"],
    ["ViT-B-16", "laion2b_s34b_b88k"],
    ["ViT-B-16", "datacomp_l_s1b_b8k"],
    ["ViT-B-16", "commonpool_l_laion_s1b_b8k"],
    ["ViT-B-16", "dfn2b"],
    ["ViT-B-16-quickgelu", "metaclip_fullcc"],
    ["ViT-B-16-plus-240", "laion400m_e31"],
    ["ViT-L-14", "openai"],
    ["ViT-L-14", "laion400m_e31"],
    ["ViT-L-14", "datacomp_xl_s13b_b90k"],
    ["ViT-L-14", "commonpool_xl_clip_s13b_b90k"],
    ["ViT-L-14-quickgelu", "metaclip_fullcc"],
    ["ViT-L-14-quickgelu", "dfn2b"],
    ["ViT-L-14-336", "openai"],
    ["ViT-H-14", "laion2b_s32b_b79k"],
    ["ViT-H-14-quickgelu", "metaclip_fullcc"],
    ["ViT-H-14-378-quickgelu", "dfn5b"],
    ["ViT-g-14", "laion2b_s12b_b42k"],
    ["ViT-bigG-14", "laion2b_s39b_b160k"],
    ["roberta-ViT-B-32", "laion2b_s12b_b32k"],
    ["xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"],
    ["convnext_base_w", "laion2b_s13b_b82k"],
    ["convnext_base_w_320", "laion_aesthetic_s13b_b82k"],
    ["convnext_large_d", "laion2b_s26b_b102k_augreg"],
    ["convnext_large_d_320", "laion2b_s29b_b131k_ft"],
    ["convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"],
    ["coca_ViT-B-32", "laion2b_s13b_b90k"],
    ["coca_ViT-L-14", "laion2b_s13b_b90k"],
    ["EVA01-g-14", "laion400m_s11b_b41k"],
    ["EVA02-B-16", "merged2b_s8b_b131k"],
    ["EVA02-L-14-336", "merged2b_s6b_b61k"],
    ["EVA02-E-14", "laion2b_s4b_b115k"],
    ["nllb-clip-base", "v1"],
    ["nllb-clip-base-siglip", "v1"]
]

# ===============================
# MNIST 类名与多模板 prompt
# ===============================
classnames = [str(i) for i in range(10)]
TEMPLATES = [
    "a photo of the handwritten digit {}",
    "a grayscale image of the number {}",
    "a drawing of the number {}",
    "an MNIST image of the digit {}",
    "a picture of the number {}",
]

# ===============================
# 构建文本特征（多模板平均）
# ===============================
@torch.no_grad()
def build_text_features(model, tokenizer):
    texts = []
    for c in classnames:
        for t in TEMPLATES:
            texts.append(t.format(c))
    text_tokens = tokenizer(texts).to(device)

    # 文本塔固定 fp32（不开 AMP）
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 还原为 [10, len(TEMPLATES), dim]，按模板平均
    text_features = text_features.view(len(classnames), len(TEMPLATES), -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# ===============================
# 评估函数（使用各自 preprocess）
# ===============================
@torch.no_grad()
def evaluate_model(model_name, pretrained):
    print(f"\n🔍 Evaluating {model_name}-{pretrained} ...")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name}-{pretrained}: {e}")
        return None

    # 为 MNIST 构建 transform：灰度转 RGB，再接模型的 preprocess
    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        preprocess,
    ])
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=64, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True
    )

    # 构建文本特征
    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer)

    correct, total = 0, 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features.float() @ text_features.float().T

        preds = torch.argmax(logits, dim=-1)
        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"✅ {model_name}-{pretrained}: {acc:.2f}%")
    return {"model": model_name, "pretrained": pretrained, "acc": acc}

# ===============================
# 主循环
# ===============================
results = []
for model_name, pretrained in MODEL_LIST:
    res = evaluate_model(model_name, pretrained)
    if res:
        results.append(res)

# 保存结果
df = pd.DataFrame(results)
df.to_csv("mnist_zero_shot_results.csv", index=False)
print("\n📊 Done! Results saved to mnist_zero_shot_results.csv")
print(df.sort_values("acc", ascending=False))
