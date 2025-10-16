import os
import csv
import glob
import torch
from torchvision import transforms
from PIL import Image
import open_clip
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ===============================
# 参数设置
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/zssd/home/weiyang/modelselection/datasets/datasets/gtsrb" 
# ===============================
# 模型列表（与原代码一致）
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
# 类别名称加载（GTSRB）
# ===============================
def load_gtsrb_classnames(root):
    signnames_csv = os.path.join(root, "signnames.csv")
    if os.path.isfile(signnames_csv):
        names = [""] * 43
        with open(signnames_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["ClassId"])
                names[cid] = row["SignName"]
        return names
    else:
        # 若没有 signnames.csv，使用默认描述
        return [
            "speed limit 20 km/h", "speed limit 30 km/h", "speed limit 50 km/h", "speed limit 60 km/h",
            "speed limit 70 km/h", "speed limit 80 km/h", "end of speed limit 80 km/h", "speed limit 100 km/h",
            "speed limit 120 km/h", "no passing", "no passing for vehicles over 3.5 metric tons",
            "right-of-way at the next intersection", "priority road", "yield", "stop", "no vehicles",
            "vehicles over 3.5 metric tons prohibited", "no entry", "general caution", "dangerous curve to the left",
            "dangerous curve to the right", "double curve", "bumpy road", "slippery road", "road narrows on the right",
            "road work", "traffic signals", "pedestrians", "children crossing", "bicycles crossing",
            "beware of ice/snow", "wild animals crossing", "end of all speed and passing limits", "turn right ahead",
            "turn left ahead", "ahead only", "go straight or right", "go straight or left", "keep right",
            "keep left", "roundabout mandatory", "end of no passing", "end no passing by vehicles over 3.5 tons", "other"
        ]

TEMPLATES = [
    "a traffic sign meaning {}.",
    "a zoomed in photo of a {} traffic sign.",
    "a centered photo of a {} traffic sign.",
    "a close up photo of a {} traffic sign."
]

# ===============================
# Dataset 定义
# ===============================
class GTSRBDataset(Dataset):
    def __init__(self, root, split="train", transform=None, include_labels=True):
        self.split = split
        self.transform = transform
        self.include_labels = include_labels

        self.train_dir = os.path.join(root, "GTSRB", "Training")
        self.test_dir = os.path.join(root, "GTSRB", "Final_Test", "Images")
        self.gt_csv = os.path.join(root, "GT-final_test.csv")

        if split == "train":
            self.samples = self._load_train()
        else:
            self.samples = self._load_test()

    def _load_train(self):
        samples = []
        for class_dir in sorted(os.listdir(self.train_dir)):
            class_path = os.path.join(self.train_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            class_id = int(class_dir)
            for ext in ("*.ppm", "*.png", "*.jpg"):
                for img_path in glob.glob(os.path.join(class_path, ext)):
                    samples.append((img_path, class_id))
        return samples

    def _load_test(self):
        if not os.path.exists(self.gt_csv):
            raise FileNotFoundError(f"Missing GT-final_test.csv at {self.gt_csv}")
        samples = []
        with open(self.gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = os.path.join(self.test_dir, row["Filename"])
                if os.path.isfile(img_path):
                    label = int(row["ClassId"])
                    samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.include_labels:
            return image, label
        return image

# ===============================
# 文本特征构建
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
    print(f"\n🔍 Evaluating {model_name}-{pretrained} on GTSRB {split} set ...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name}-{pretrained}: {e}")
        return None

    classnames = load_gtsrb_classnames(DATA_ROOT)
    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer, classnames)

    dataset = GTSRBDataset(DATA_ROOT, split=split, transform=preprocess)
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
# 主执行逻辑
# ===============================
if __name__ == "__main__":
    results = []
    for model_name, pretrained in MODEL_LIST:
        res = evaluate_model(model_name, pretrained, split="test")
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("gtsrb_zero_shot_results.csv", index=False)
    print("\n📊 Done! Results saved to gtsrb_zero_shot_results.csv")
    print(df.sort_values("acc", ascending=False))
