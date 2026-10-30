#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGE-M3 检索模型效果评测脚本

功能:
  1. 用 FlagEmbedding 加载本地 BGE-M3 模型 (pytorch_model.bin)
  2. 一次编码同时得到 dense / sparse / colbert 三种表示
  3. 分别评测 dense、sparse、hybrid、colbert 四种检索模式
  4. 指标: Hit@1/3/10, MRR, Recall@10, nDCG@10, 并统计耗时

用法:
  python eval_bge_m3.py                       # 使用内置评测集, 全部模式
  python eval_bge_m3.py --modes dense,hybrid  # 只测部分模式
  python eval_bge_m3.py --no-colbert          # 跳过 colbert 模式 (省内存/时间)
  python eval_bge_m3.py --corpus my_corpus.json --queries my_queries.json --qrels my_qrels.json

外部数据文件格式 (JSON):
  corpus.json : [{"id": "d0", "text": "..."}, ...]
  queries.json: [{"id": "q0", "text": "..."}, ...]
  qrels.json  : {"q0": ["d0", "d20"], ...}   # 每个查询对应的相关文档 id 列表
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:  # 兼容旧版 FlagEmbedding 的导入路径
    from FlagEmbedding.BGE_M3 import BGEM3FlagModel


# ---------------------------------------------------------------------------
# 内置评测集: 30 篇中英混合文档 + 11 个查询 (含跨语言标注, 体现多语言检索能力)
# ---------------------------------------------------------------------------
CORPUS = [
    {"id": "d0", "text": "中兴通讯是全球领先的综合通信解决方案提供商，在5G基站、核心网和光传输领域拥有强大实力。"},
    {"id": "d1", "text": "5G网络相比4G具有更高的传输速率、更低的时延和更大的连接密度，可支持万物互联。"},
    {"id": "d2", "text": "机器学习是人工智能的核心分支，通过数据训练模型来实现预测和分类任务。"},
    {"id": "d3", "text": "深度学习利用多层神经网络自动提取数据特征，在图像识别和语音识别领域取得突破。"},
    {"id": "d4", "text": "自然语言处理（NLP）研究计算机与人类语言之间的交互，包括机器翻译、问答系统和情感分析。"},
    {"id": "d5", "text": "信息检索是从大规模文档集合中找到与用户查询相关文档的技术，搜索引擎是典型应用。"},
    {"id": "d6", "text": "词嵌入（Word Embedding）将词语映射到低维稠密向量空间，语义相近的词在空间中距离更近。"},
    {"id": "d7", "text": "向量数据库专门存储和检索高维向量，广泛用于语义搜索和推荐系统。"},
    {"id": "d8", "text": "量子计算利用量子比特的叠加和纠缠特性，在某些特定问题上可能远超经典计算机。"},
    {"id": "d9", "text": "电动汽车以动力电池为能源，具有零尾气排放和能源利用效率高的优点。"},
    {"id": "d10", "text": "锂电池是当前电动汽车最主流的动力电池，能量密度高、循环寿命长。"},
    {"id": "d11", "text": "全球气候变暖主要由温室气体排放引起，导致海平面上升和极端天气频发。"},
    {"id": "d12", "text": "可再生能源包括太阳能、风能和水能，是应对气候变化的重要手段。"},
    {"id": "d13", "text": "唐朝是中国历史上国力强盛的朝代，文化繁荣，丝绸之路贸易兴盛。"},
    {"id": "d14", "text": "足球世界杯是国际足联主办的世界性足球赛事，每四年举办一次。"},
    {"id": "d15", "text": "Python是一种解释型、面向对象的高级编程语言，语法简洁，广泛应用于数据科学。"},
    {"id": "d16", "text": "网络安全涉及保护计算机系统和网络免受攻击、破坏和未授权访问。"},
    {"id": "d17", "text": "数据加密技术通过算法将明文转换为密文，保障信息传输和存储的机密性。"},
    {"id": "d18", "text": "云计算通过互联网按需提供计算资源、存储和服务，降低企业IT成本。"},
    {"id": "d19", "text": "智能手机集通信、摄影、娱乐和移动支付等功能于一体，已成为人们生活的必需品。"},
    {"id": "d20", "text": "ZTE Corporation is a leading global provider of telecommunications equipment, offering 5G base stations and optical networks."},
    {"id": "d21", "text": "Machine learning algorithms learn patterns from data to make predictions and classifications."},
    {"id": "d22", "text": "Deep neural networks with multiple layers have achieved breakthrough performance in computer vision."},
    {"id": "d23", "text": "Natural language processing enables computers to understand and generate human language."},
    {"id": "d24", "text": "Search engines retrieve relevant documents from large collections based on user queries."},
    {"id": "d25", "text": "Word embeddings map words into dense vector spaces where semantically similar words are close."},
    {"id": "d26", "text": "Quantum computers leverage superposition and entanglement to solve certain problems faster than classical machines."},
    {"id": "d27", "text": "Electric vehicles powered by lithium-ion batteries produce zero tailpipe emissions."},
    {"id": "d28", "text": "Renewable energy sources such as solar and wind power help mitigate climate change."},
    {"id": "d29", "text": "Python is a high-level programming language widely used in data science and web development."},
    {"id":"d30","text":"中兴 5G 属于全球第一梯队，基站、核心网端到端能力成熟，Massive MIMO、节能、行业专网、5G-A 是强项；整体仅次于华为，强于爱立信、诺基亚，短板主要在高端自研芯片、海外市场受限"},
    {"id": "d31","text": "中兴  zte chisi"}
]

QUERIES = [
    {"id": "q0", "text": "中兴通讯的5G技术怎么样？"},
    {"id": "q1", "text": "什么是机器学习？"},
    {"id": "q2", "text": "How do deep neural networks work?"},
    {"id": "q3", "text": "搜索引擎如何找到相关文档？"},
    {"id": "q4", "text": "What are word embeddings?"},
    {"id": "q5", "text": "量子计算为什么比经典计算机快？"},
    {"id": "q6", "text": "Which battery is used in electric vehicles?"},
    {"id": "q7", "text": "可再生能源有哪些，能应对气候变化吗？"},
    {"id": "q8", "text": "Python适合做什么？"},
    {"id": "q9", "text": "What is natural language processing?"},
    {"id": "q10", "text": "如何保护网络信息安全？"},
]

# qrels 是人工/规则标注的"这个查询应该返回哪些文档"
QRELS = {
    "q0": ["d0", "d20","d30"],
    "q1": ["d2", "d21"],
    "q2": ["d3", "d22"],
    "q3": ["d5", "d24", "d6"],
    "q4": ["d6", "d25"],
    "q5": ["d8", "d26"],
    "q6": ["d10", "d27"],
    "q7": ["d12", "d28"],
    "q8": ["d15", "d29"],
    "q9": ["d4", "d23"],
    "q10": ["d16", "d17"],
}

# 各模式的 compute_score 权重 [dense, sparse, colbert]
MODE_WEIGHTS = {
    "dense": [1.0, 0.0, 0.0],
    "sparse": [0.0, 1.0, 0.0],
    "colbert": [0.0, 0.0, 1.0],
    "hybrid": [1.0, 0.3, 1.0],  # FlagEmbedding 官方默认混合权重
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(qrels, rankings, top_k):
    """基于二值相关性计算检索指标。

    qrels   : {qid: [相关doc_id, ...]}
    rankings: {qid: [doc_id, ...]}  按分数从高到低
    """
    ks = [1, 3, 10]
    n = len(qrels)
    hit = {k: 0 for k in ks}
    mrr_sum = rec_sum = ndcg_sum = 0.0
    for qid, rel in qrels.items():
        rel = set(rel)
        ranked = [d for d, _ in rankings[qid][:top_k]]  # 只取 doc id
        for k in ks:
            if rel & set(ranked[:k]):
                hit[k] += 1
        for i, d in enumerate(ranked):  # MRR
            if d in rel:
                mrr_sum += 1.0 / (i + 1)
                break
        rec_sum += len(rel & set(ranked)) / len(rel)  # Recall@top_k
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), top_k)))
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0
    return {
        "hit@1": hit[1] / n,
        "hit@3": hit[3] / n,
        "hit@10": hit[10] / n,
        "mrr": mrr_sum / n,
        f"recall@{top_k}": rec_sum / n,
        f"ndcg@{top_k}": ndcg_sum / n,
    }


def print_sample(queries, corpus, doc_text, rankings, modes, n_q=2, depth=3):
    """打印前几个查询的 Top-k 结果, 便于人工查看检索质量。"""
    for m in modes:
        print(f"\n--- [{m}] 样例结果 (前 {n_q} 个查询, Top-{depth}) ---")
        for q in queries[:n_q]:
            qid = q["id"]
            print(f"  Q: {q['text'][:40]}")
            for i, (did, score) in enumerate(rankings[m][qid][:depth]):
                print(f"    {i + 1}. [{did}] {score:.4f}  {doc_text[did][:50]}")


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 检索效果评测")
    parser.add_argument("--model-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge-m3"),
                        help="本地模型目录 (默认: 脚本同级的 bge-m3/)")
    parser.add_argument("--corpus", help="外部语料 JSON 文件路径 (可选)")
    parser.add_argument("--queries", help="外部查询 JSON 文件路径 (可选)")
    parser.add_argument("--qrels", help="外部标注 JSON 文件路径 (可选)")
    parser.add_argument("--modes", default="all",
                        help="评测模式, 逗号分隔: dense,sparse,hybrid,colbert 或 all (默认 all)")
    parser.add_argument("--no-colbert", action="store_true",
                        help="完全跳过 colbert 向量编码 (省内存/时间, hybrid 退化为 dense+sparse)")
    parser.add_argument("--top-k", type=int, default=10, help="召回深度 (默认 10)")
    parser.add_argument("--batch-size", type=int, default=12, help="编码 batch size (默认 12)")
    parser.add_argument("--device", default="cpu", help="推理设备 cpu/cuda (默认 cpu)")
    args = parser.parse_args()

    # ---------- 数据准备 ----------
    corpus = load_json(args.corpus) if args.corpus else CORPUS
    queries = load_json(args.queries) if args.queries else QUERIES
    qrels = load_json(args.qrels) if args.qrels else QRELS
    doc_text = {d["id"]: d["text"] for d in corpus}
    assert set(qrels) == {q["id"] for q in queries}, "qrels 与 queries 的 id 不一致"

    # ---------- 模型文件检查 ----------
    for f in ["pytorch_model.bin", "config.json", "sentencepiece.bpe.model"]:
        if not os.path.exists(os.path.join(args.model_path, f)):
            print(f"[警告] 缺少文件 {f}, 模型可能无法加载")
    if not args.no_colbert:
        for f in ["sparse_linear.pt", "colbert_linear.pt"]:
            if not os.path.exists(os.path.join(args.model_path, f)):
                print(f"[警告] 缺少 {f}, sparse/colbert 模式将不可用")

    # ---------- 模式解析 ----------
    if args.modes == "all":
        modes = ["dense", "sparse", "hybrid"] + ([] if args.no_colbert else ["colbert"])
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODE_WEIGHTS]
    if not modes:
        print("错误: --modes 未指定有效模式"); sys.exit(1)
    need_colbert = "colbert" in modes or "hybrid" in modes
    if args.no_colbert:
        modes = [m for m in modes if m != "colbert"]
        MODE_WEIGHTS["hybrid"] = [1.0, 0.3, 0.0]  # hybrid 退化为 dense+sparse

    # ---------- 加载模型 ----------
    print(f"加载模型: {args.model_path} (device={args.device}) ...")
    t0 = time.time()
    model = BGEM3FlagModel(args.model_path, use_fp16=(args.device != "cpu"), device=args.device)
    print(f"模型加载完成, 耗时 {time.time() - t0:.1f}s")

    # ---------- 编码 (一次编码, 各模式复用) ----------
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    q_ids = [q["id"] for q in queries]
    q_texts = [q["text"] for q in queries]

    print(f"编码 {len(doc_texts)} 篇文档 ...")
    t0 = time.time()
    d_emb = model.encode(doc_texts, batch_size=args.batch_size, max_length=8192,
                         return_dense=True, return_sparse=True, return_colbert_vecs=need_colbert)
    t_doc = time.time() - t0
    print(f"  文档编码 {t_doc:.1f}s ({t_doc / len(doc_texts):.2f}s/doc)")

    print(f"编码 {len(q_texts)} 个查询 ...")
    t0 = time.time()
    q_emb = model.encode(q_texts, batch_size=args.batch_size, max_length=8192,
                         return_dense=True, return_sparse=True, return_colbert_vecs=need_colbert)
    t_q = time.time() - t0
    print(f"  查询编码 {t_q:.1f}s ({t_q / len(q_texts):.2f}s/query)")

    # ---------- 三种基础打分矩阵 (FlagEmbedding 官方计算方式) ----------
    q_dense = torch.tensor(np.asarray(q_emb["dense_vecs"]), dtype=torch.float32)
    d_dense = torch.tensor(np.asarray(d_emb["dense_vecs"]), dtype=torch.float32)
    dense_scores = np.asarray(model.model.compute_dense_score(q_dense, d_dense))

    sparse_scores = np.asarray(
        model.compute_lexical_matching_score(q_emb["lexical_weights"], d_emb["lexical_weights"]))

    if need_colbert:
        def pad_colbert(vecs):
            max_len = max(v.shape[0] for v in vecs)
            padded = torch.zeros(len(vecs), max_len, vecs[0].shape[1])
            mask = torch.zeros(len(vecs), max_len, dtype=torch.long)
            for i, v in enumerate(vecs):
                padded[i, :v.shape[0]] = torch.tensor(v, dtype=torch.float32)
                mask[i, :v.shape[0]] = 1
            return padded, mask

        q_col, q_mask = pad_colbert(q_emb["colbert_vecs"])
        d_col, _ = pad_colbert(d_emb["colbert_vecs"])
        colbert_scores = np.asarray(model.model.compute_colbert_score(q_col, d_col, q_mask=q_mask))
    else:
        colbert_scores = np.zeros_like(dense_scores)

    base_scores = [dense_scores, sparse_scores, colbert_scores]

    # ---------- 各模式打分 + 排序 ----------
    results = {}
    rankings = {}
    for m in modes:
        t0 = time.time()
        w = MODE_WEIGHTS[m]
        # 与官方 compute_score 一致: 参与模式的加权平均
        active = [(s, w[i]) for i, s in enumerate(base_scores) if w[i] > 0]
        scores = sum(s * wi for s, wi in active) / sum(wi for _, wi in active)
        order = np.argsort(-scores, axis=1)
        rankings[m] = {qid: [(doc_ids[j], float(scores[i, j])) for j in order[i]]
                       for i, qid in enumerate(q_ids)}
        metrics = compute_metrics(qrels, rankings[m], args.top_k)
        metrics["score_time_s"] = time.time() - t0
        results[m] = metrics

    # ---------- 输出 ----------
    header = f"{'mode':<10}" + "".join(f"{k:>10}" for k in ["hit@1", "hit@3", "hit@10", "mrr",
                                                            f"recall@{args.top_k}", f"ndcg@{args.top_k}"])
    print("\n==================== 评测结果 ====================")
    print(f"语料: {len(corpus)} 篇文档 | 查询: {len(queries)} 个 | 模式数: {len(modes)}")
    print(header)
    print("-" * len(header))
    for m in modes:
        r = results[m]
        print(f"{m:<10}" + "".join(f"{r[k]:>10.4f}" for k in ["hit@1", "hit@3", "hit@10", "mrr",
                                                               f"recall@{args.top_k}", f"ndcg@{args.top_k}"]))
    print(f"\n编码耗时: 文档 {t_doc:.1f}s | 查询 {t_q:.1f}s | 单查询平均 {t_q / len(q_texts):.2f}s")
    print("打分耗时(各模式): " + ", ".join(f"{m}={results[m]['score_time_s']:.2f}s" for m in modes))

    print_sample(queries, corpus, doc_text, rankings, modes, n_q=2, depth=3)

    out = {"model_path": args.model_path, "top_k": args.top_k,
           "n_docs": len(corpus), "n_queries": len(queries),
           "encode_time": {"docs_s": t_doc, "queries_s": t_q}, "results": results}
    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
