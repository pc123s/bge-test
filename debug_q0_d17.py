"""诊断: 查询 q0 '中兴通讯的5G技术怎么样' 为什么能查到 d17 (数据加密...)。

打印各模式下 q0 的文档排名, 并拆解 d17 的 sparse 共同 token 贡献。
"""
import sys
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from eval_bge_m3 import CORPUS, QUERIES, QRELS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载模型 (device={device}) ...")
model = BGEM3FlagModel("bge-m3", use_fp16=(device != "cpu"), device=device)

q = next(q for q in QUERIES if q["id"] == "q0")
docs = CORPUS
texts = [q["text"]] + [d["text"] for d in docs]
emb = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)

# --- 三种基础打分 ---
q_dense = torch.tensor(np.asarray(emb["dense_vecs"][0]), dtype=torch.float32).unsqueeze(0)
d_dense = torch.tensor(np.asarray(emb["dense_vecs"][1:]), dtype=torch.float32)
dense = np.asarray(model.model.compute_dense_score(q_dense, d_dense))[0]

sparse = np.asarray(model.compute_lexical_matching_score(
    [emb["lexical_weights"][0]], emb["lexical_weights"][1:]))[0]

# colbert: pad 到批量内最大长度
def pad_colbert(vecs):
    max_len = max(v.shape[0] for v in vecs)
    padded = torch.zeros(len(vecs), max_len, vecs[0].shape[1])
    mask = torch.zeros(len(vecs), max_len, dtype=torch.long)
    for i, v in enumerate(vecs):
        padded[i, :v.shape[0]] = torch.tensor(v, dtype=torch.float32)
        mask[i, :v.shape[0]] = 1
    return padded, mask

q_col, q_mask = pad_colbert(emb["colbert_vecs"][:1])
d_col, _ = pad_colbert(emb["colbert_vecs"][1:])
colbert = np.asarray(model.model.compute_colbert_score(q_col, d_col, q_mask=q_mask))[0]

hybrid = (1.0 * dense + 0.3 * sparse + 1.0 * colbert) / 2.3

rel = set(QRELS["q0"])
d17_idx = next(i for i, d in enumerate(docs) if d["id"] == "d17")

print(f"\n查询: {q['text']!r}  相关文档: {sorted(rel)}\n")
for name, scores in [("dense", dense), ("sparse", sparse), ("colbert", colbert), ("hybrid", hybrid)]:
    order = np.argsort(-scores)
    print(f"--- {name} 排名 (top-8) ---")
    for r, i in enumerate(order[:8], 1):
        did = docs[i]["id"]
        mark = "*" if did in rel else ("<-- d17" if did == "d17" else " ")
        print(f"  {r}. [{did}] {scores[i]:.3f} {mark}  {docs[i]['text'][:40]}")
    if "d17" not in [docs[i]["id"] for i in order[:8]]:
        rank = int(np.where(order == d17_idx)[0][0]) + 1
        print(f"  ... d17 排第 {rank}/{len(docs)} 位, 分数 {scores[d17_idx]:.3f}")
    print()

# --- 拆解 d17 的 sparse 贡献 ---
qi, di = 0, next(i for i, d in enumerate(docs) if d["id"] == "d17")
q_lw, d_lw = emb["lexical_weights"][qi], emb["lexical_weights"][di + 1]


def dec(tid):
    return model.tokenizer.decode([int(tid)])


common = {t: (q_lw[t], d_lw[t]) for t in q_lw if t in d_lw}
print(f"=== d17 的 sparse 拆解 (总分 {sparse[di]:.3f}) ===")
print(f"查询词权 top: " + "  ".join(
    f"{dec(t)!r}={w:.2f}" for t, w in sorted(q_lw.items(), key=lambda kv: -kv[1])[:6]))
print(f"d17 词权 top: " + "  ".join(
    f"{dec(t)!r}={w:.2f}" for t, w in sorted(d_lw.items(), key=lambda kv: -kv[1])[:6]))
print("共同 token 贡献:")
for t, (qw, dw) in sorted(common.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
    print(f"  {dec(t)!r:16s} {qw:.2f} x {dw:.2f} = {qw*dw:.3f}")
