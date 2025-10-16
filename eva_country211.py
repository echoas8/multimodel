import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import open_clip
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = Path("/zssd/home/weiyang/modelselection/datasets/datasets/country211/country211")
batch_size = 64
num_workers = 8  # DataLoader 并行 worker 数量

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
# ISO 国家代码到全称映射（完整 Country211）
# -----------------------------
code2name = {
    'AD': 'Andorra', 'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AG': 'Antigua and Barbuda', 'AI': 'Anguilla',
    'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AQ': 'Antarctica', 'AR': 'Argentina', 'AS': 'American Samoa',
    'AT': 'Austria', 'AU': 'Australia', 'AW': 'Aruba', 'AX': 'Åland Islands', 'AZ': 'Azerbaijan',
    'BA': 'Bosnia and Herzegovina', 'BB': 'Barbados', 'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso',
    'BG': 'Bulgaria', 'BH': 'Bahrain', 'BI': 'Burundi', 'BJ': 'Benin', 'BL': 'Saint Barthélemy', 'BM': 'Bermuda',
    'BN': 'Brunei Darussalam', 'BO': 'Bolivia', 'BQ': 'Bonaire', 'BR': 'Brazil', 'BS': 'Bahamas', 'BT': 'Bhutan',
    'BV': 'Bouvet Island', 'BW': 'Botswana', 'BY': 'Belarus', 'BZ': 'Belize', 'CA': 'Canada', 'CC': 'Cocos Islands',
    'CD': 'Democratic Republic of the Congo', 'CF': 'Central African Republic', 'CG': 'Congo', 'CH': 'Switzerland',
    'CI': "Côte d'Ivoire", 'CK': 'Cook Islands', 'CL': 'Chile', 'CM': 'Cameroon', 'CN': 'China', 'CO': 'Colombia',
    'CR': 'Costa Rica', 'CU': 'Cuba', 'CV': 'Cabo Verde', 'CW': 'Curaçao', 'CX': 'Christmas Island', 'CY': 'Cyprus',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DJ': 'Djibouti', 'DK': 'Denmark', 'DM': 'Dominica', 'DO': 'Dominican Republic',
    'DZ': 'Algeria', 'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egypt', 'EH': 'Western Sahara', 'ER': 'Eritrea',
    'ES': 'Spain', 'ET': 'Ethiopia', 'FI': 'Finland', 'FJ': 'Fiji', 'FM': 'Micronesia', 'FO': 'Faroe Islands',
    'FR': 'France', 'GA': 'Gabon', 'GB': 'United Kingdom', 'GD': 'Grenada', 'GE': 'Georgia', 'GF': 'French Guiana',
    'GG': 'Guernsey', 'GH': 'Ghana', 'GI': 'Gibraltar', 'GL': 'Greenland', 'GM': 'Gambia', 'GN': 'Guinea',
    'GP': 'Guadeloupe', 'GQ': 'Equatorial Guinea', 'GR': 'Greece', 'GT': 'Guatemala', 'GU': 'Guam', 'GW': 'Guinea-Bissau',
    'GY': 'Guyana', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croatia', 'HT': 'Haiti', 'HU': 'Hungary',
    'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel', 'IM': 'Isle of Man', 'IN': 'India', 'IQ': 'Iraq',
    'IR': 'Iran', 'IS': 'Iceland', 'IT': 'Italy', 'JE': 'Jersey', 'JM': 'Jamaica', 'JO': 'Jordan', 'JP': 'Japan',
    'KE': 'Kenya', 'KG': 'Kyrgyzstan', 'KH': 'Cambodia', 'KI': 'Kiribati', 'KM': 'Comoros', 'KN': 'Saint Kitts and Nevis',
    'KP': 'North Korea', 'KR': 'South Korea', 'KW': 'Kuwait', 'KY': 'Cayman Islands', 'KZ': 'Kazakhstan', 'LA': 'Laos',
    'LB': 'Lebanon', 'LC': 'Saint Lucia', 'LI': 'Liechtenstein', 'LK': 'Sri Lanka', 'LR': 'Liberia', 'LS': 'Lesotho',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'LY': 'Libya', 'MA': 'Morocco', 'MC': 'Monaco',
    'MD': 'Moldova', 'ME': 'Montenegro', 'MF': 'Saint Martin', 'MG': 'Madagascar', 'MH': 'Marshall Islands',
    'MK': 'North Macedonia', 'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia', 'MO': 'Macao', 'MP': 'Northern Mariana Islands',
    'MQ': 'Martinique', 'MR': 'Mauritania', 'MS': 'Montserrat', 'MT': 'Malta', 'MU': 'Mauritius', 'MV': 'Maldives',
    'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique', 'NA': 'Namibia', 'NC': 'New Caledonia',
    'NE': 'Niger', 'NF': 'Norfolk Island', 'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Netherlands', 'NO': 'Norway',
    'NP': 'Nepal', 'NR': 'Nauru', 'NU': 'Niue', 'NZ': 'New Zealand', 'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru',
    'PF': 'French Polynesia', 'PG': 'Papua New Guinea', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland',
    'PM': 'Saint Pierre and Miquelon', 'PN': 'Pitcairn', 'PR': 'Puerto Rico', 'PS': 'Palestine', 'PT': 'Portugal',
    'PW': 'Palau', 'PY': 'Paraguay', 'QA': 'Qatar', 'RE': 'Réunion', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia',
    'RW': 'Rwanda', 'SA': 'Saudi Arabia', 'SB': 'Solomon Islands', 'SC': 'Seychelles', 'SD': 'Sudan', 'SE': 'Sweden',
    'SG': 'Singapore', 'SH': 'Saint Helena', 'SI': 'Slovenia', 'SJ': 'Svalbard and Jan Mayen', 'SK': 'Slovakia',
    'SL': 'Sierra Leone', 'SM': 'San Marino', 'SN': 'Senegal', 'SO': 'Somalia', 'SR': 'Suriname', 'SS': 'South Sudan',
    'ST': 'São Tomé and Príncipe', 'SV': 'El Salvador', 'SX': 'Sint Maarten', 'SY': 'Syria', 'SZ': 'Eswatini',
    'TC': 'Turks and Caicos Islands', 'TD': 'Chad', 'TF': 'French Southern Territories', 'TG': 'Togo', 'TH': 'Thailand',
    'TJ': 'Tajikistan', 'TK': 'Tokelau', 'TL': 'Timor-Leste', 'TM': 'Turkmenistan', 'TN': 'Tunisia', 'TO': 'Tonga',
    'TR': 'Turkey', 'TT': 'Trinidad and Tobago', 'TV': 'Tuvalu', 'TZ': 'Tanzania', 'UA': 'Ukraine', 'UG': 'Uganda',
    'US': 'United States', 'UY': 'Uruguay', 'UZ': 'Uzbekistan', 'VA': 'Vatican City', 'VC': 'Saint Vincent and the Grenadines',
    'VE': 'Venezuela', 'VG': 'Virgin Islands (British)', 'VI': 'Virgin Islands (U.S.)', 'VN': 'Vietnam', 'VU': 'Vanuatu',
    'WF': 'Wallis and Futuna', 'WS': 'Samoa', 'YE': 'Yemen', 'YT': 'Mayotte', 'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'
}

# -----------------------------
# 批量评估函数
# -----------------------------
def evaluate_model(model_name, pretrained, class_names, test_path):
    print(f"\n🔹 Evaluating {model_name}-{pretrained} ...")

    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ✅ 改进后的 Prompt
    prompts = [
        f"a photo took in country {code2name.get(name, name)}" for name in class_names
    ]
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc=f"{model_name}-{pretrained}"):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ {model_name}-{pretrained} Accuracy: {acc:.4f}")
    return acc

# -----------------------------
# 批量评估所有模型
# -----------------------------
dummy_model, _, dummy_transform = open_clip.create_model_and_transforms(model_list[0][0], pretrained=model_list[0][1])
dummy_dataset = datasets.ImageFolder(data_root / "test", transform=dummy_transform)
class_names = dummy_dataset.classes
print(f"Loaded {len(dummy_dataset)} test images with {len(class_names)} classes")

results = []
for model_name, pretrained in model_list:
    acc = evaluate_model(model_name, pretrained, class_names, data_root / "test")
    if acc is not None:
        results.append({"model_name": f"{model_name}-{pretrained}", "accuracy": acc})

# -----------------------------
# 保存 CSV
# -----------------------------
df = pd.DataFrame(results)
csv_path = "country211_vlm_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 All results saved to {csv_path}")
print(df)
