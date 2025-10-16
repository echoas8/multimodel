import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import open_clip
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ===============================
# 参数设置
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/zssd/home/weiyang/modelselection/datasets/datasets/stl10/stl10_binary"

# ===============================
# 模型列表
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
# 读取 STL10 类别名称
# ===============================
def load_stl10_classnames(root):
    fname = os.path.join(root, "class_names.txt")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Missing class_names.txt in {root}")
    with open(fname, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

TEMPLATES = [
    "a photo of a {}.",
    "an image of a {}.",
    "a picture of a {}.",
    "a photo showing a {}."
]

# ===============================
# STL10 Binary Dataset
# ===============================
class STL10BinaryDataset(Dataset):
    def __init__(self, root, split="test", transform=None, include_labels=True):
        assert split in ["train", "test"], "Only 'train' or 'test' supported"
        self.root = root
        self.transform = transform
        self.include_labels = include_labels

        # 加载二进制文件
        if split == "train":
            data_file = os.path.join(root, "train_X.bin")
            label_file = os.path.join(root, "train_y.bin")
        else:
            data_file = os.path.join(root, "test_X.bin")
            label_file = os.path.join(root, "test_y.bin")

        self.data = self._read_images(data_file)
        self.labels = self._read_labels(label_file) if include_labels else None

    def _read_images(self, filepath):
        with open(filepath, "rb") as f:
            data = np.fromfile(f, dtype=np.uint8)
        # 每张图像 96x96x3
        data = np.reshape(data, (-1, 3, 96, 96))
        # 转换成 [H, W, C]
        data = np.transpose(data, (0, 2, 3, 1))
        return data

    def _read_labels(self, filepath):
        with open(filepath, "rb") as f:
            labels = np.fromfile(f, dtype=np.uint8)
        # STL10 标签从 1 开始 → 转为 0-based
        labels = labels - 1
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        if self.include_labels:
            return img, int(self.labels[idx])
        return img

# ===============================
# 构建文本特征（多模板平均）
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
def evaluate_model(model_name, pretrained, split="test"):
    print(f"\n🔍 Evaluating {model_name}-{pretrained} on STL10 {split} set ...")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name}-{pretrained}: {e}")
        return None

    classnames = load_stl10_classnames(DATA_ROOT)
    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer, classnames)

    dataset = STL10BinaryDataset(DATA_ROOT, split=split, transform=preprocess)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

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
        res = evaluate_model(model_name, pretrained, split="test")
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("stl10_zero_shot_results.csv", index=False)
    print("\n📊 Done! Results saved to stl10_zero_shot_results.csv")
    print(df.sort_values("acc", ascending=False))
