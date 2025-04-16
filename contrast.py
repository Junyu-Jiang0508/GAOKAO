import os
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm

# 设置输出目录
os.makedirs("output", exist_ok=True)

# Deepseek API 参数设置 (你需要填写自己的API KEY)
API_KEY = "sk-af036f229df94036826cc54071741677"
API_URL = "https://api.deepseek.com/chat/completions"

# 加载数据
df = pd.read_csv("output/gold_standard.csv")

# 标签编码
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])

# 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label_id'], test_size=0.2, random_state=42
)

# 通过Few-shot Prompt进行文本分类
def deepseek_classify(text, label_list):
    prompt = f"""
你是一名高考文本分类专家，请将下面的新闻文本准确地分类到以下类别之一：

类别：{', '.join(label_list)}

文本：{text}

请只返回你预测的类别名称，不要有其他任何解释。
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",  # 或具体模型名如 deepseek-llm-base-zh
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # 确保确定性输出
        "max_tokens": 10,
        "top_p": 1
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    predicted_label = result['choices'][0]['message']['content'].strip()

    return predicted_label

# 对验证集进行预测
preds = []
label_list = label_encoder.classes_.tolist()

for text in tqdm(X_val, desc="Deepseek 模型预测中"):
    try:
        pred_label = deepseek_classify(text, label_list)
        preds.append(pred_label)
    except Exception as e:
        print(f"预测时发生错误：{e}")
        preds.append("未知类别")

# 将预测标签转为数字编码
preds_encoded = label_encoder.transform([
    label if label in label_list else label_list[0] for label in preds
])

# 真实标签
true_labels = y_val.tolist()

# 计算分类报告并保存
report = classification_report(
    true_labels,
    preds_encoded,
    labels=list(range(len(label_list))),
    target_names=label_list,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("output/deepseek_api_classification_report.csv", encoding="utf-8-sig")

# 保存混淆矩阵
cm = confusion_matrix(true_labels, preds_encoded, labels=list(range(len(label_list))))
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
cm_df.to_csv("output/deepseek_api_confusion_matrix.csv", encoding="utf-8-sig")

print("✅ Deepseek API 调用测试完成，分类报告和混淆矩阵已保存到 output 文件夹。")
