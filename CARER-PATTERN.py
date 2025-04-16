import pandas as pd
from collections import defaultdict, Counter
import jieba
import networkx as nx
import json

# 加载数据
df = pd.read_csv("output/final_cleaned_gaokao_texts.csv")

# 构建共现图（滑动窗口）
window_size = 3
edges = defaultdict(int)

for text in df['text']:
    tokens = list(jieba.cut(text))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            w1, w2 = tokens[i], tokens[j]
            if w1 != w2:
                edges[(w1, w2)] += 1

# 创建无向图
G = nx.Graph()
for (w1, w2), weight in edges.items():
    G.add_edge(w1, w2, weight=weight)

# 计算中心性和聚类系数
eigen_centrality = nx.eigenvector_centrality_numpy(G)
clustering_coef = nx.clustering(G)

# 筛选 connector 和 subject words
connector_words = {word for word, score in eigen_centrality.items() if score > 0.05}
subject_words = {word for word, score in clustering_coef.items() if score > 0.3}

# 抽取基础情绪 patterns
patterns = []

def extract_patterns(tokens, cw_set, sw_set):
    for i in range(len(tokens) - 2):
        tri = tokens[i:i+3]
        if tri[0] in cw_set and tri[1] in sw_set:
            patterns.append((tri[0], '*'))
        if tri[0] in sw_set and tri[1] in cw_set and tri[2] in sw_set:
            patterns.append(('*', tri[1], '*'))

for text in df['text']:
    tokens = list(jieba.cut(text))
    extract_patterns(tokens, connector_words, subject_words)

# 统计频率，保留高频 patterns
pattern_freq = Counter(patterns)
filtered_patterns = {p: f for p, f in pattern_freq.items() if f > 2}

# 保存为 JSON
filtered_patterns_str_keys = {"|".join(p): f for p, f in filtered_patterns.items()}

# 保存为 JSON
with open("patterns_basic.json", "w", encoding="utf-8") as f:
    json.dump(filtered_patterns_str_keys, f, ensure_ascii=False, indent=2)

