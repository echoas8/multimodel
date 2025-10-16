import os
import torch
import open_clip
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

# 设置GPU================================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据路径================================
DATA_ROOT = "/home/weiyang/modelselection/datasets/datasets/voc2007/VOCdevkit/VOC2007"  # ← 改成你的VOC2007路径
JPEG_DIR = os.path.join(DATA_ROOT, "JPEGImages")
ANNOT_DIR = os.path.join(DATA_ROOT, "Annotations")

# 模型列表================================
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

# VOC类别名================================
BASE_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 姿态集合================================
POSES = ["Left", "Right", "Frontal", "Unspecified"]

# 类别+姿态组合================================
classnames = []
for c in BASE_CLASSES:
    for p in POSES:
        cname = f"{c} facing {p.lower()}" if p != "Unspecified" else c
        classnames.append(cname)

# 文本模板================================
TEMPLATES = [
    "a photo of a {}.",
    "an image containing a {}.",
    "a picture showing a {}.",
    "a realistic photo of a {}."
]

# 数据集定义================================
class VOCPoseDataset(Dataset):
    def __init__(self, annot_dir, jpeg_dir, transform=None):
        self.annot_dir = annot_dir
        self.jpeg_dir = jpeg_dir
        self.transform = transform
        self.samples = []

        xml_files = [f for f in os.listdir(annot_dir) if f.endswith(".xml")]
        for xml_file in xml_files:
            xml_path = os.path.join(annot_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find("filename").text
            img_path = os.path.join(jpeg_dir, filename)
            if not os.path.exists(img_path):
                continue

            label_set = set()
            for obj in root.findall("object"):
                cls_name = obj.find("name").text.strip().lower()
                pose = obj.find("pose").text.strip().capitalize() if obj.find("pose") is not None else "Unspecified"
                if pose in ["Left", "Right", "Frontal"]:
                    combo = f"{cls_name} facing {pose.lower()}"
                else:
                    combo = cls_name
                label_set.add(combo)
            self.samples.append((img_path, list(label_set)))

        print(f"📁 Loaded {len(self.samples)} annotated images with class+pose labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, labels

# 文本特征构建================================
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

# 模型评估================================
@torch.no_grad()
def evaluate_model(model_name, pretrained):
    print(f"\n🔍 Evaluating {model_name}-{pretrained} on VOC2007 (class+pose) multi-label test ...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name}-{pretrained}: {e}")
        return None

    dataset = VOCPoseDataset(ANNOT_DIR, JPEG_DIR, transform=preprocess)

    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, labels

    loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
    )


    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = build_text_features(model, tokenizer)

    total, correct_per_class = 0, torch.zeros(len(classnames))
    total_per_class = torch.zeros(len(classnames))

    for images, labels_list in tqdm(loader, leave=False):
        images = images.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features.float() @ text_features.float().T  # [B, Nclass]
        preds = (logits > 0.25).int().cpu()  # 阈值判断为存在该类别，可调整

        for i, gt_labels in enumerate(labels_list):
            gt_indices = [classnames.index(lbl) for lbl in gt_labels if lbl in classnames]
            gt_vec = torch.zeros(len(classnames))
            gt_vec[gt_indices] = 1
            total_per_class += gt_vec
            correct_per_class += (preds[i] * gt_vec)
            total += 1

    per_class_acc = (correct_per_class / total_per_class.clamp(min=1e-5)) * 100
    mean_acc = per_class_acc.mean().item()

    print(f"✅ {model_name}-{pretrained}: mAP-like mean accuracy {mean_acc:.2f}%")
    return {"model": model_name, "pretrained": pretrained, "mean_acc": mean_acc}

# 主程序================================
if __name__ == "__main__":
    results = []
    for model_name, pretrained in MODEL_LIST:
        res = evaluate_model(model_name, pretrained)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    out_csv = "voc2007_pose_multilabel_zero_shot_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📊 Done! Results saved to {out_csv}")
    print(df.sort_values("mean_acc", ascending=False))
