import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_empty_state_and_disabled_input(self):
        with patch('hw3.retrieval.load_index',side_effect=FileNotFoundError('找不到向量索引；請先執行 03_build_index.py')):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            self.assertIn('找不到向量索引',app.info[0].value)
            self.assertTrue(app.checkbox[0].value)

    def test_real_controls_with_fake_model(self):
        from test_core import FakeLLM,FakeRetriever
        with patch('hw3.retrieval.load_index',return_value=([],[],{'created_at':'測試'})),patch('hw3.llm.LocalLLM',FakeLLM),patch('hw3.retrieval.Retriever',FakeRetriever):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value('吃飯速度如何調整？').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['turns']),1)
            self.assertIn('測試模型回答',[x.value for x in app.markdown])
            app.checkbox[0].uncheck().run()
            app.chat_input[0].set_value('請再解釋一下').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state['turns'][-1]['retrieved_this_turn'])
            next(b for b in app.button if b.label=='整理本次對話重點').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['turns'][-1]['mode'],'summary')
            next(b for b in app.button if b.label=='清除對話').click().run()
            self.assertEqual(len(app.session_state['turns']),0)

    def test_selecting_old_question_updates_evidence_without_losing_history(self):
        """切換是瀏覽舊回答，不能偷偷重問模型或把兩輪來源混在一起。"""
        from test_core import SOURCE
        first={'question':'第一題','answer':'第一個回答','raw_answer':'第一個回答','sources':[{**SOURCE,'summary':'僅屬於第一輪的摘要'}],'mode':'question','retrieved_this_turn':True}
        second={'question':'第二題','answer':'第二個回答','raw_answer':'第二個回答','sources':[{**SOURCE,'url':'https://www.healthnews.com.tw/article/456','summary':'僅屬於第二輪的摘要'}],'mode':'question','retrieved_this_turn':True}
        with patch('hw3.retrieval.load_index',return_value=([],[],{'created_at':'測試'})):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'))
            app.session_state['turns']=[first,second]
            app.session_state['selected_turn']=1
            app.run()
            self.assertFalse(app.exception)
            self.assertIn('僅屬於第二輪的摘要',[x.value for x in app.markdown])
            next(b for b in app.button if b.key=='history_0').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['selected_turn'],0)
            self.assertEqual(len(app.session_state['turns']),2)
            content=[x.value for x in app.markdown]
            self.assertIn('第一個回答',content)
            self.assertIn('僅屬於第一輪的摘要',content)
            self.assertNotIn('僅屬於第二輪的摘要',content)
            next(b for b in app.button if b.label=='清除對話').click().run()
            self.assertEqual(app.session_state['turns'],[])
            self.assertIsNone(app.session_state['selected_turn'])
