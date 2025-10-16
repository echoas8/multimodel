import os
import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# =====================
# 配置部分
# =====================
ROOT = "/zssd/home/weiyang/modelselection/datasets/datasets/dtd"
LABELS_PATH = os.path.join(ROOT, "labels/labels_joint_anno.txt")
TEST_FILES = [f"test{i}.txt" for i in range(1, 4)]
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_workers = 4

# 模型列表
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

# =====================
# 读取标签
# =====================
img_to_labels = {}
with open(LABELS_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        img_to_labels[parts[0]] = parts[1:]

all_labels = sorted({l for labs in img_to_labels.values() for l in labs})
label_to_idx = {l: i for i, l in enumerate(all_labels)}
print(f"✅ 共 {len(all_labels)} 个标签。")

# =====================
# Dataset 定义
# =====================
class DTDataset(Dataset):
    def __init__(self, root, list_paths, img_to_labels, transform=None):
        self.root = os.path.join(root, "dtd/images")
        self.imgs = []
        for lp in list_paths:
            with open(os.path.join(root, "labels", lp), "r") as f:
                self.imgs.extend([x.strip() for x in f.readlines()])
        self.img_to_labels = img_to_labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rel_path = self.imgs[idx]
        img_path = os.path.join(self.root, rel_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.img_to_labels[rel_path]
        return image, labels

# =====================
# 自定义 collate_fn
# =====================
def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels

# =====================
# DataLoader
# =====================
def get_loader(transform):
    dataset = DTDataset(ROOT, TEST_FILES, img_to_labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    return loader

# =====================
# 评估函数
# =====================
def evaluate_clip_accuracy(model, tokenizer, label_texts, loader):
    text_inputs = tokenizer(label_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, pred_idx in enumerate(preds):
                pred_label = all_labels[pred_idx]
                if pred_label in targets[i]:
                    correct += 1
                total += 1
    return correct / total

# =====================
# 主循环 & 保存 CSV
# =====================
results = []

for name, pretrained in model_list:
    print(f"\n🚀 Evaluating {name} ({pretrained}) ...")
    try:
        # 每个模型加载官方 transform（保证图片统一大小 + Tensor + Normalize）
        model, _, transform = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(name)

        loader = get_loader(transform)
        label_texts = [f"a photo showing the texture of {l}" for l in all_labels]

        acc = evaluate_clip_accuracy(model, tokenizer, label_texts, loader)
        print(f"  🔹 Accuracy = {acc:.4f}")

        results.append({"model_name": f"{name}-{pretrained}", "accuracy": acc})
    except Exception as e:
        print(f"❌ {name} ({pretrained}) 评估失败: {e}")

# 保存 CSV
df = pd.DataFrame(results)
csv_path = "dtd_zero_shot_accuracy.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 所有模型结果已保存到 {csv_path}")
