# bge-test

[BGE-M3](https://huggingface.co/BAAI/bge-m3) 检索模型效果评测脚本。

## 环境准备

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 下载模型（除 pytorch_model.bin 以外的文件按需下载；PyTorch 推理需要 pytorch_model.bin）：

   ```bash
   hf download BAAI/bge-m3 --local-dir bge-m3
   ```

   模型文件放在脚本同级的 `bge-m3/` 目录（已加入 .gitignore，不入库）。

## 运行评测

```bash
# 完整评测: dense / sparse / hybrid / colbert 四种模式
python eval_bge_m3.py

# 只测部分模式
python eval_bge_m3.py --modes dense,hybrid

# 跳过 colbert (省内存/时间, hybrid 退化为 dense+sparse)
python eval_bge_m3.py --no-colbert

# 使用自己的数据 (JSON 格式)
python eval_bge_m3.py --corpus my_corpus.json --queries my_queries.json --qrels my_qrels.json
```

外部数据格式：

- `corpus.json` : `[{"id": "d0", "text": "..."}, ...]`
- `queries.json`: `[{"id": "q0", "text": "..."}, ...]`
- `qrels.json`  : `{"q0": ["d0", "d20"], ...}`（每个查询对应的相关文档 id 列表）

## 指标

Hit@1 / Hit@3 / Hit@10、MRR、Recall@10、nDCG@10，以及编码耗时；结果同时保存到 `eval_results.json`。
