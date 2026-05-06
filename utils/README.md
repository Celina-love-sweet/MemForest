
# 🛠️ 1. Replacement of the retrieval mechanism

We have integrated our proposed **Anchor-Guided Propagation Retrieval(AGPR)** mechanism into the open-source **Mem0** and **M3-Agent** frameworks. The original retrieval mechanisms provided by the frameworks can be found in `MemForest/utils/Mem0` and `MemForest/utils/M3-Agent`, which you can replace at the following code locations:

### 1.1. 🧠 Mem0

You can replace the code in `MemForest/Mem0/evaluation/src/memzero/search.py` with the implementation from `MemForest/utils/Mem0` to restore the original retrieval mechanism.

### 1.2. 🤖 M3-Agent

You can replace the code in `MemForest/M3_Agent/m3_agent/control.py` and `MemForest/M3_Agent/mmagent/retrieve.py` with the implementation from `MemForest/utils/M3_Agent` to restore the original retrieval mechanisms.

---

# ✂ 2. Segmentation of Generated Memories

Since the LongMemEval and PersonaMem datasets generate a large amount of memory, we recommend splitting the datasets first and then compressing them using `MemForest/Mem0/evaluation/MemForest.py`. This can be done by running the following code:

```bash
python split_mem_store_by_sample.py
python merge_mem_store_subs_sqlite.py
```

Please note that you need to modify the `DEFAULT_SRC_STORE` and `DEFAULT_DST_ROOT` parameters in `split_mem_store_by_sample.py`, as well as the `--subs-root` and `--dst-store` parameters in `merge_mem_store_subs_sqlite.py`, to control the storage of split memories and the path of the newly merged files. Additionally, during compression, you should adjust the `DEFAULT_MEM_STORE_PATH`, `--start-idx`, and `--end-idx` parameters in `MemForest/Mem0/evaluation/MemForest.py`.
