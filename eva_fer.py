import os
import torch
from torchvision import transforms
from PIL import Image
import open_clip
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ===============================
# 参数设置
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/zssd/home/weiyang/modelselection/datasets/datasets/fer2013"

# 模型列表（保留主干模型）
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
# 类别名与模板（FER2013专用）
# ===============================
classnames = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

TEMPLATES = [
    "a photo of a person showing {} expression.",
    "a photo of a face that looks {}.",
    "a facial expression of being {}.",
    "a person looking {}.",
    "a close-up of a human face that looks {}."
]

# ===============================
# 自定义 Dataset for FER2013
# ===============================
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None, include_labels=True):
        df = pd.read_csv(csv_file)
        if "Usage" in df.columns:
            df = df[df["Usage"].str.lower().str.contains(split)]
        self.include_labels = include_labels
        self.transform = transform
        self.images = df["pixels"].tolist()
        self.labels = df["emotion"].tolist() if include_labels and "emotion" in df.columns else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pixels = np.array(self.images[idx].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, int(self.labels[idx])
        return img

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
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.view(len(classnames), len(TEMPLATES), -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# ===============================
# 评估函数
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

    dataset = FER2013Dataset(
        os.path.join(DATA_ROOT, "train.csv"),
        split="train",
        transform=preprocess
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer)

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
    results = []
    for model_name, pretrained in MODEL_LIST:
        res = evaluate_model(model_name, pretrained)
        if res:
            results.append(res)

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("fer2013_transfer.csv", index=False)
    print("\n📊 Done! Results saved to fer2013_zero_shot_results.csv")
    print(df.sort_values("acc", ascending=False))
