import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 超参数
# -------------------------------
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 8
LEARNING_RATE = 1e-3

# -------------------------------
# 加载数据
# -------------------------------
df_label = pd.read_csv("output/gaokao_labeled.csv")
df_weights = pd.read_csv("pattern_weights.csv")
pattern_vector_map = dict(zip(df_weights['pattern'], df_weights.iloc[:, 1:].values))

# ✅ 显式指定情绪类别（8类）
emotions = ['joy', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation']
label_encoder = LabelEncoder()
df_label = df_label[df_label['label'].isin(emotions)]  # 只保留8类
df_label['label_id'] = label_encoder.fit_transform(df_label['label'])

# -------------------------------
# pattern 匹配函数
# -------------------------------
def extract_patterns(text, patterns):
    matched = []
    for pattern in patterns:
        parts = pattern.split("|")
        regex = re.escape(parts[0]) if parts[0] != "*" else ".+?"
        for t in parts[1:]:
            regex += ".+" + (re.escape(t) if t != "*" else ".+?")
        if re.search(regex, text):
            matched.append(pattern)
    return matched

def get_text_vector(text):
    matched = extract_patterns(text, pattern_vector_map.keys())
    if not matched:
        return np.zeros(EMBED_DIM)
    vectors = [pattern_vector_map[p] for p in matched if p in pattern_vector_map]
    return np.mean(vectors, axis=0) if vectors else np.zeros(EMBED_DIM)

# -------------------------------
# 构建训练集
# -------------------------------
X = np.array([get_text_vector(t) for t in df_label['text']])
y = df_label['label_id'].values

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EmotionDataset(X_val, y_val), batch_size=BATCH_SIZE)

# -------------------------------
# 模型定义
# -------------------------------
class CARERCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * input_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)

model = CARERCNN(EMBED_DIM, len(emotions))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 模型训练
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            pred_labels = torch.argmax(preds, dim=1)
            correct += (pred_labels == yb).sum().item()
            total += len(yb)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - val_acc: {acc:.4f}")

# -------------------------------
# 模型保存
# -------------------------------
torch.save(model.state_dict(), "carer_emotion_cnn.pt")
print("✅ 模型训练完成，保存为 carer_emotion_cnn.pt")
