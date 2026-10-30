"""查看 BGE-M3 sparse 词权重 / dense 分数, 解释 sparse 为何效果差。

用法: python inspect_sparse.py
"""
import sys
import torch
from FlagEmbedding import BGEM3FlagModel

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK, 强制 UTF-8

sys.path.insert(0, ".")
from eval_bge_m3 import CORPUS, QUERIES, QRELS  # 复用内置评测集

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载模型 (device={device}) ...")
model = BGEM3FlagModel("bge-m3", use_fp16=(device != "cpu"), device=device)

texts = [q["text"] for q in QUERIES] + [d["text"] for d in CORPUS]
emb = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)

n_q = len(QUERIES)
q_sparse = emb["lexical_weights"][:n_q]
d_sparse = emb["lexical_weights"][n_q:]
q_dense = emb["dense_vecs"][:n_q]
d_dense = emb["dense_vecs"][n_q:]


def decode(tid):
    return model.tokenizer.decode([int(tid)])


def show_query_weights(qid, top=12):
    qi = next(i for i, q in enumerate(QUERIES) if q["id"] == qid)
    lw = sorted(q_sparse[qi].items(), key=lambda kv: -kv[1])[:top]
    print(f"\n查询 [{qid}] {QUERIES[qi]['text']!r} 的 top-{top} 词权重:")
    for tid, w in lw:
        print(f"  {decode(tid)!r:20s}  weight={w:.3f}")


def compare(qid, doc_ids):
    qi = next(i for i, q in enumerate(QUERIES) if q["id"] == qid)
    print(f"\n=== 查询 [{qid}] {QUERIES[qi]['text']!r} 对各文档打分 ===")
    for did in doc_ids:
        di = next(i for i, d in enumerate(CORPUS) if d["id"] == did)
        # sparse: 找出共同 token 及贡献
        common = {t: (q_sparse[qi][t], d_sparse[di][t])
                  for t in q_sparse[qi] if t in d_sparse[di]}
        sp = sum(qw * dw for qw, dw in common.values())
        # dense: 余弦相似度 (编码时已归一化, 点积即余弦)
        de = float(q_dense[qi] @ d_dense[di])
        print(f"\n  [{did}] {CORPUS[di]['text'][:48]}...")
        print(f"    sparse={sp:.3f}   dense={de:.3f}")
        if common:
            parts = [f"{decode(t)!r}({qw:.2f}x{dw:.2f}={qw*dw:.3f})"
                     for t, (qw, dw) in sorted(common.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:8]]
            print("    共同token贡献: " + "  ".join(parts))
        else:
            print("    共同token: 无 -> sparse 只能打 0 分")


# q0: 跨语言对 (中文查询 vs 英文相关文档) 最能体现 sparse 的短板
show_query_weights("q0")
compare("q0", ["d0", "d20", "d30", "d31", "d5"])

show_query_weights("q10")
compare("q10", ["d16", "d17", "d5"])

# q2: 英文查询, 相关文档一中一英
compare("q2", ["d3", "d22", "d2"])
