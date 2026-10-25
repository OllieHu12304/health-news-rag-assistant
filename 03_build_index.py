"""將地端摘要編碼為 E5 向量，保存 NumPy 索引、文章 ID 與版本資訊。"""
import json
from hw3.retrieval import Embedder,build_index

if __name__=='__main__':
    # 真正的編碼、ID 對照與版本檢查在 hw3/retrieval.py；此處是命令列入口。
    # 印出的 manifest 可用來確認文章數、向量維度與模型版本。
    print(json.dumps(build_index(Embedder()),ensure_ascii=False,indent=2))
