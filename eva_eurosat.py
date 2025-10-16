import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import open_clip
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = Path("/zssd/home/weiyang/modelselection/datasets/datasets/eurosat/2750")

# -----------------------------
# 读取 EuroSAT
# -----------------------------
def load_eurosat(data_root):
    class_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    images, labels = [], []
    for idx, cname in enumerate(class_names):
        img_dir = data_root / cname
        for img_path in img_dir.glob("*.jpg"):
            images.append(str(img_path))
            labels.append(idx)
    return images, labels, class_names

images, labels, class_names = load_eurosat(data_root)
print(f"Loaded EuroSAT dataset: {len(images)} images, {len(class_names)} classes")

# -----------------------------
# Dataset 定义
# -----------------------------
class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# 模型列表（原样）
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
# 批量评估函数（多模板 prompts）
# -----------------------------
def evaluate_model_multi_prompt(model_name, pretrained, batch_size=64):
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name} ({pretrained}): {e}")
        return None

    dataset = EuroSATDataset(images, labels, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 多模板 prompts
    templates = [
        "a satellite photo of a {}",
        "aerial imagery showing {}",
        "a remote sensing image of {}",
        "a top-down view of {}",
        "a high-resolution image of {}"
    ]
    
    # 对每个类别生成文本特征并平均
    all_text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for cname in class_names:
            text_feats_per_template = []
            for template in templates:
                prompt = template.format(cname.replace("_", " "))
                text_tokens = tokenizer([prompt]).to(device)
                feat = model.encode_text(text_tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                text_feats_per_template.append(feat)
            avg_feat = torch.stack(text_feats_per_template, dim=0).mean(dim=0)
            avg_feat /= avg_feat.norm(dim=-1, keepdim=True)  # 再归一化
            all_text_features.append(avg_feat)
        text_features = torch.cat(all_text_features, dim=0)

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
    acc = evaluate_model_multi_prompt(model_name, pretrained)
    if acc is not None:
        results.append({"model_name": f"{model_name}-{pretrained}", "accuracy": acc})

# -----------------------------
# 保存 CSV
# -----------------------------
df = pd.DataFrame(results)
csv_path = "eurosat_vlm_eval_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 所有结果已保存到: {csv_path}")
print(df)
