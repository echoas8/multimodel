import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io
from pathlib import Path
import open_clip
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 配置部分
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

data_root = Path("/zssd/home/weiyang/modelselection/datasets/datasets/cars")
annos = scipy.io.loadmat(data_root / "cars_annos.mat")
class_names = [x[0] for x in annos["class_names"][0]]

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
# 定义 Cars 数据集
# -----------------------------
class CarsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# -----------------------------
# 构建测试集 DataLoader（batch 方式）
# -----------------------------
test_indices = [i for i, x in enumerate(annos["annotations"][0]) if x[-1][0][0] == 1]
test_images = []
test_labels = []

for idx in test_indices:
    a = annos["annotations"][0][idx]
    rel_path = a[0][0]
    label = int(a[5][0][0]) - 1  # matlab index starts at 1
    test_images.append(data_root / rel_path)
    test_labels.append(label)

print(f"Loaded {len(test_images)} test images.")

# -----------------------------
# 定义评估函数（batch 方式）
# -----------------------------
def evaluate_model(model_name, pretrained, batch_size=64):
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load {model_name} ({pretrained}): {e}")
        return None

    # 构建 DataLoader
    test_dataset = CarsDataset(test_images, test_labels, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 文本特征
    prompts = [f"a photo of a {name}, a type of car" for name in class_names]
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    # 批量评估
    for images, labels in tqdm(test_loader, desc=f"{model_name}-{pretrained}"):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ {model_name}-{pretrained}: {acc:.4f}")
    return acc

# -----------------------------
# 主循环：批量评估所有模型
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
csv_path = "car_vlm_eval_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 所有结果已保存到: {csv_path}")
print(df)
