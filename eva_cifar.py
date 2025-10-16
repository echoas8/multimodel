import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
from pathlib import Path
import open_clip
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = Path("/zssd/home/weiyang/modelselection/datasets/datasets/cifar-100-python")

# -----------------------------
# 读取 CIFAR-100
# -----------------------------
def load_cifar100(split="test"):
    path = data_root / split
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW->NHWC
    labels = entry[b"fine_labels"]
    return images, labels

def load_meta():
    path = data_root / "meta"
    with open(path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    fine_names = [x.decode("utf-8") for x in meta[b"fine_label_names"]]
    return fine_names

test_images, test_labels = load_cifar100("test")
fine_names = load_meta()
print(f"Loaded CIFAR-100 test set: {len(test_images)} images, {len(fine_names)} classes")

# -----------------------------
# Dataset 定义
# -----------------------------
class CIFAR100Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# 模型列表
# -----------------------------
model_list = [
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

# -----------------------------
# 批量评估函数
# -----------------------------
def evaluate_model(model_name, pretrained, batch_size=128):
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name} ({pretrained}): {e}")
        return None

    dataset = CIFAR100Dataset(test_images, test_labels, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 文本特征
    prompts = [f"a photo of a {name}" for name in fine_names]
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    for imgs, labs in tqdm(loader, desc=f"{model_name}-{pretrained}"):
        imgs = imgs.to(device)
        labs = labs.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)

    acc = correct / total
    print(f"✅ {model_name}-{pretrained}: {acc:.4f}")
    return acc

# -----------------------------
# 批量评估所有模型
# -----------------------------
results = []
for model_name, pretrained in model_list:
    acc = evaluate_model(model_name, pretrained)
    if acc is not None:
        results.append({"model_name": f"{model_name}-{pretrained}", "accuracy": acc})

# -----------------------------
# 保存 CSV
# -----------------------------
df = pd.DataFrame(results)
csv_path = "cifar100_vlm_eval_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 所有结果已保存到: {csv_path}")
print(df)
