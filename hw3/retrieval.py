# 向量檢索流程：摘要 → E5 編碼 → 正規化向量 → 與問題向量比相似度。
import json
import os
import numpy as np
from .config import EMBED_PATH, EMBED_ID, INDEX_PATH, MANIFEST_PATH, DB_PATH
from .db import articles,corpus_hash,now


class Embedder:
    def __init__(self, path=EMBED_PATH):
        from transformers import AutoTokenizer, AutoModel
        import torch
        if not (path/'config.json').exists():
            raise FileNotFoundError('找不到 embedding 模型。請先執行 00_download_models.py')
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        # eval 關閉訓練模式；embedding 使用 CPU，生成模型則交給 MLX。
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True).eval().to('cpu')
        source=path/'hw3_model_source.json'
        self.revision=json.loads(source.read_text())['revision'] if source.exists() else None

    def encode(self, texts, query=False):
        import torch
        if not texts: raise ValueError('沒有可向量化文字')
        output=[]
        # 分批處理以限制記憶體；E5 要求問題加 query:，文件加 passage: 前綴。
        # 512 是含特殊標記的 token 上限；更長的摘要在這一步會截斷。
        for i in range(0,len(texts),8):
            batch=self.tokenizer([('query: ' if query else 'passage: ')+s for s in texts[i:i+8]],padding=True,truncation=True,max_length=512,return_tensors='pt')
            # 只做推論，不保留梯度計算所需的資料。
            with torch.inference_mode():
                hidden=self.model(**batch).last_hidden_state
                # hidden 形狀為 [批次, token數, 向量維度]。
                # padding 只為湊齊批次長度，不應影響整段文字的平均向量。
                mask=batch['attention_mask'].unsqueeze(-1).bool()
                # 遮罩平均：先把 padding 向量歸零，再除以有效 token 數。
                pooled=hidden.masked_fill(~mask,0).sum(1)/mask.sum(1)
                # L2 正規化讓每個向量長度為 1，因此之後內積就是 cosine similarity。
                output.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
        return np.concatenate(output).astype(np.float32)


def build_index(embedder, db_path=DB_PATH, index_path=INDEX_PATH, manifest_path=MANIFEST_PATH):
    rows=articles(db_path,summarized=True)
    if not rows: raise ValueError('沒有摘要；請先執行 01_crawl.py 與 02_summarize.py')
    if len(rows)!=len(articles(db_path)): raise ValueError('仍有文章缺少摘要，請完成 02_summarize.py 再建立索引')
    # 作業要求對「摘要」向量化；不是把標題或全文直接當成索引輸入。
    vectors=embedder.encode([r['summary'] for r in rows])
    # manifest 是索引的說明檔：記錄模型版本、維度、語料指紋等重現資訊。
    manifest={'created_at':now(),'count':len(rows),'model_id':EMBED_ID,'revision':embedder.revision,
              'max_tokens':512,'pooling':'attention-mask mean','normalization':'L2','query_prefix':'query: ','passage_prefix':'passage: ',
              'corpus_sha256':corpus_hash(rows),'dimension':int(vectors.shape[1])}
    index_path.parent.mkdir(exist_ok=True,parents=True)
    tmp=index_path.with_suffix('.tmp.npz')
    # ids[i] 對應 vectors[i]，查到向量第 i 列才知道它是哪一篇新聞。
    # 先寫暫存檔再取代正式檔，避免中途停止留下半份向量檔。
    np.savez_compressed(tmp,ids=np.array([r['id'] for r in rows],dtype=np.int64),vectors=vectors)
    os.replace(tmp,index_path)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def load_index(db_path=DB_PATH,index_path=INDEX_PATH,manifest_path=MANIFEST_PATH):
    if not index_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('找不到向量索引；請先執行 03_build_index.py')
    manifest=json.loads(manifest_path.read_text())
    rows=articles(db_path,summarized=True)
    # 新增文章、修改摘要等都可能使原索引失效；寧可要求重建，也不默用舊資料。
    if manifest['corpus_sha256']!=corpus_hash(rows) or len(rows)!=len(articles(db_path)):
        raise ValueError('新聞或摘要已變更；請完成摘要後重新執行 03_build_index.py')
    # 不允許載入 pickle 物件；這份索引只需要數字陣列。
    with np.load(index_path,allow_pickle=False) as data:
        ids=data['ids'].copy(); vectors=data['vectors'].copy()
    # 同時檢查文章順序、矩陣形狀及數值是否有限，防止來源錯配或損毀。
    if ids.tolist()!=[r['id'] for r in rows] or vectors.shape!=(len(rows),manifest['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('向量索引與新聞 ID 不一致；請重建索引')
    return rows,vectors,manifest


class Retriever:
    def __init__(self, embedder=None):
        self.rows,self.vectors,self.manifest=load_index()
        self.embedder=embedder or Embedder()
        if self.manifest['revision']!=self.embedder.revision:
            raise ValueError('embedding 模型版本與索引不一致，請重建索引')

    def search(self, query, top_k=3, min_score=0.83):
        if not query.strip(): raise ValueError('問題不可空白')
        # 網頁服務可能運作很久；每次查詢都重讀並驗證索引，避免沿用過期資料。
        rows,vectors,manifest=load_index()
        if manifest['revision']!=self.embedder.revision:
            raise ValueError('embedding 模型版本已改變，請重新載入應用')
        self.rows,self.vectors,self.manifest=rows,vectors,manifest
        # [文章數, 維度] @ [維度] 得到每篇文章對這個問題的相似度。
        scores=self.vectors@self.embedder.encode([query],query=True)[0]
        # argsort 預設由小到大，因此對負分數排序，取得由高到低的前 top_k 名。
        order=np.argsort(-scores)[:top_k]
        # 候選須同時達到最低門檻，且距最高分不超過 0.02，減少混入弱相關文章。
        # 這是開發階段設定的篩選規則；分數不是模型回答正確的機率。
        return [{**self.rows[i],'score':round(float(scores[i]),6)} for i in order if scores[i]>=max(min_score,float(scores.max())-0.02)]
