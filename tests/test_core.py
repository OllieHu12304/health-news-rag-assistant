import tempfile
import unittest
from pathlib import Path
import numpy as np
from hw3.db import upsert,articles,save_summary
from hw3.retrieval import build_index,load_index
from hw3.chat import ChatEngine,SUMMARY_PROMPT

ARTICLE={'title':'吃飯速度','url':'https://www.healthnews.com.tw/article/123','content':'有效正文'*100,'published_at':'2026-10-20T09:00:00+08:00','category':'健康保健'}
SOURCE={**ARTICLE,'id':1,'summary':'吃飯慢一點。','score':0.9}

class FakeEmbedding:
    revision='test-only'
    def encode(self,texts,query=False): return np.tile(np.array([1.,0.],dtype=np.float32),(len(texts),1))

class FakeLLM:
    def __init__(self): self.messages=[]
    def generate(self,messages,max_tokens):
        self.messages=messages
        return {'text':'測試模型回答','raw':'測試模型回答'}

class FakeRetriever:
    def __init__(self): self.queries=[]
    def search(self,query): self.queries.append(query); return [dict(SOURCE)]

class DatabaseIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.db=root/'news.db';self.idx=root/'idx.npz';self.manifest=root/'manifest.json'
    def tearDown(self): self.tmp.cleanup()
    def test_dedup_and_update_invalidates_summary(self):
        self.assertTrue(upsert(ARTICLE,self.db));self.assertFalse(upsert(ARTICLE,self.db))
        save_summary(1,'摘要',{},self.db)
        self.assertEqual(len(articles(self.db)),1)
        upsert({**ARTICLE,'content':'變更正文'*100},self.db)
        self.assertIsNone(articles(self.db)[0]['summary'])
    def test_index_mapping_and_stale_corpus(self):
        upsert(ARTICLE,self.db);save_summary(1,'摘要',{},self.db)
        build_index(FakeEmbedding(),self.db,self.idx,self.manifest)
        rows,vectors,_=load_index(self.db,self.idx,self.manifest)
        self.assertEqual(rows[0]['url'],ARTICLE['url']);self.assertEqual(vectors.shape,(1,2))
        save_summary(1,'已修改摘要',{},self.db)
        with self.assertRaisesRegex(ValueError,'變更'): load_index(self.db,self.idx,self.manifest)
    def test_empty_index_and_partial_summaries_rejected(self):
        with self.assertRaisesRegex(ValueError,'沒有摘要'): build_index(FakeEmbedding(),self.db,self.idx,self.manifest)
        upsert(ARTICLE,self.db);save_summary(1,'摘要',{},self.db)
        upsert({**ARTICLE,'url':'https://www.healthnews.com.tw/article/456'},self.db)
        with self.assertRaisesRegex(ValueError,'缺少摘要'): build_index(FakeEmbedding(),self.db,self.idx,self.manifest)

class ChatTests(unittest.TestCase):
    def setUp(self):
        self.llm=FakeLLM();self.retriever=FakeRetriever();self.engine=ChatEngine(self.llm,self.retriever,evidence_loader=lambda: [])
    def test_checkbox_off_first_turn_no_fabricated_sources(self):
        turn=self.engine.answer('問題',[],retrieve=False)
        self.assertFalse(turn['sources']);self.assertFalse(self.retriever.queries);self.assertFalse(self.llm.messages)
    def test_checkbox_off_retains_evidence(self):
        first=self.engine.answer('吃飯速度如何調整',[])
        second=self.engine.answer('再說一次',[first],retrieve=False)
        self.assertEqual(second['sources'][0]['url'],SOURCE['url'])
        self.assertEqual(len(self.retriever.queries),1)
        self.assertIn('摘要：吃飯慢一點',self.llm.messages[0]['content'])
        self.assertFalse(second['retrieved_this_turn'])
    def test_summary_preserves_both_questions_without_new_search(self):
        first=self.engine.answer('第一個不同問題',[])
        second=self.engine.answer('第二個獨立且詳細的不同問題',[first])
        turn=self.engine.answer(SUMMARY_PROMPT,[first,second])
        self.assertEqual(len(self.retriever.queries),2)
        self.assertEqual(turn['mode'],'summary');self.assertFalse(turn['retrieved_this_turn'])
        self.assertIn(first['question'],str(self.llm.messages));self.assertIn(second['question'],str(self.llm.messages))
        self.assertEqual(len(turn['sources']),1)
    def test_anaphoric_query_and_validation(self):
        first=self.engine.answer('吃飯速度與減重有什麼關係？',[])
        self.engine.answer('那要怎麼做？',[first])
        self.assertIn(first['question'],self.retriever.queries[-1])
        with self.assertRaises(ValueError): self.engine.answer('',[])
        with self.assertRaises(ValueError): self.engine.answer(SUMMARY_PROMPT,[])

class EvidenceTests(unittest.TestCase):
    def test_original_evidence_in_prompt_but_full_text_not_exported(self):
        llm=FakeLLM()
        engine=ChatEngine(llm,FakeRetriever(),evidence_loader=lambda: [dict(SOURCE)])
        turn=engine.answer('吃飯速度如何調整？',[])
        self.assertIn('有效正文',llm.messages[0]['content'])
        self.assertNotIn('content',turn['sources'][0])
        self.assertNotIn('summary_meta',turn['sources'][0])
