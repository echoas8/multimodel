import os
import torch
from torchvision import transforms
from PIL import Image
import open_clip
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ===============================
# 基本配置
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/home/weiyang/modelselection/datasets/datasets/dmlab"

# ===============================
# 类别名（从文件读取）
# ===============================
classnames = []
with open(os.path.join(DATA_ROOT, "classname.txt"), "r", encoding="utf-8") as f:
    for line in f:
        cname = line.strip()
        if cname:
            classnames.append(cname)

print(f"📚 Loaded {len(classnames)} classes from classname.txt")

# ===============================
# 模板（可根据 DMLab 语义微调）
# ===============================
TEMPLATES = [
    "{}"
]

# ===============================
# 模型列表（可按需删减）
# ===============================
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
# 自定义 Dataset for DMLab
# ===============================
class DMLabDataset(Dataset):
    def __init__(self, root, txt_file, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        with open(os.path.join(root, txt_file), "r", encoding="utf-8") as f:
            for line in f:
                path, label = line.strip().split()
                # 图像实际路径位于 root/images/...
                img_path = os.path.join(root, path)
                self.samples.append((img_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Failed to open {img_path}: {e}")
            # 若图片损坏则跳过（返回全黑图）
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================
# 构建文本特征
# ===============================
@torch.no_grad()
def build_text_features(model, tokenizer, classnames):
    texts = []
    for c in classnames:
        for t in TEMPLATES:
            texts.append(t.format(c))
    text_tokens = tokenizer(texts).to(device)
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.view(len(classnames), len(TEMPLATES), -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# ===============================
# 评估函数
# ===============================
@torch.no_grad()
def evaluate_model(model_name, pretrained, split_txt):
    print(f"\n🔍 Evaluating {model_name}-{pretrained} on {split_txt} ...")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name}-{pretrained}: {e}")
        return None

    dataset = DMLabDataset(
        DATA_ROOT,
        txt_file=split_txt,
        transform=preprocess
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer, classnames)

    correct, total = 0, 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
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
if __name__ == "__main__":
    SPLIT_TXT = "test.txt"  # 可改为 train800.txt / val200.txt 等
    results = []
    for model_name, pretrained in MODEL_LIST:
        res = evaluate_model(model_name, pretrained, SPLIT_TXT)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    out_csv = f"dmlab_zero_shot_results_{SPLIT_TXT.replace('.txt','')}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📊 Done! Results saved to {out_csv}")
    print(df.sort_values("acc", ascending=False))
