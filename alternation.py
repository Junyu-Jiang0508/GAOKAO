import os
import openai
import pandas as pd
from tqdm import tqdm

# === OpenAI 或 DeepSeek API Key 设置（建议通过环境变量）===
openai.api_key = os.getenv("sk-c01e4875c4fb4f9eb26353cd9bd05d86")
openai.api_base = "https://api.deepseek.com/v1"  # 如果你用的是 DeepSeek

# === 模型名称 ===
model_name = "deepseek-chat"  # 也可以用 gpt-3.5-turbo 或 gpt-4

# === 输入输出路径 ===
input_csv = "input.csv"
output_csv = "corrected_output.csv"

# === 调用模型进行段落校正 ===
def correct_paragraph(paragraph):
    if not paragraph.strip():
        return ""

    system_prompt = (
        "你是OCR文本纠错助手。请在不改变原文结构和语义的基础上，仅对识别错误导致不通顺的句子或明显错别字进行校正。"
        "不要进行润色、扩写、缩写或重写，只对错误之处进行最小修改。"
    )

    user_prompt = f"原始段落：\n{paragraph}\n\n请返回纠正后的段落（不要添加解释）："

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ 处理失败：{e}")
        return "[处理失败] " + paragraph

# === 主程序 ===
def process_csv():
    df = pd.read_csv(input_csv)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理段落"):
        filename = row['文件名']
        text = str(row['识别文本'])

        # 按段落切分（你也可以换成 text.split("。") 细分）
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        corrected = [correct_paragraph(p) for p in paragraphs]
        corrected_text = '\n'.join(corrected)

        results.append([filename, corrected_text])

    df_out = pd.DataFrame(results, columns=["文件名", "纠正后文本"])
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n🎉 纠正完毕，结果保存至：{output_csv}")

# === 启动 ===
if __name__ == "__main__":
    process_csv()
