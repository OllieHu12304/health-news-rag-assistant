"""兩個不同問題後整理對話，保存實際模型輸出與來源，不改寫答案。"""
import json
import platform
import time
from importlib.metadata import version
from hw3.config import ROOT
from hw3.llm import LocalLLM
from hw3.retrieval import Retriever
from hw3.chat import ChatEngine,SUMMARY_PROMPT,sources_markdown
from hw3.db import now

if __name__=='__main__':
    # 每次示範建立獨立的時間目錄，保留不同實驗輸出，不覆蓋前次結果。
    folder=ROOT/'results'/('demo_'+now().replace(':','').replace('+','_'))
    folder.mkdir(parents=True,exist_ok=False)
    llm=LocalLLM();retriever=Retriever();engine=ChatEngine(llm,retriever)
    # turns 就是對話歷史；後一輪會拿到模型自己前一輪的真實回答。
    turns=[]
    questions=['吃飯速度與減重有什麼關係？上班族可以怎麼調整？','低 FODMAP 飲食的目的與執行方式是什麼？']
    start=time.monotonic()
    # 前兩輪問不同主題；第三輪整理前文，ChatEngine 會辨識為不新增檢索的要求。
    for question in questions+[SUMMARY_PROMPT]:
        print(question,flush=True)
        turn=engine.answer(question,turns,retrieve=True)
        turns.append(turn)
        print(turn['answer']+'\n'+sources_markdown(turn)+'\n',flush=True)
        # 每輪即保存 JSON，包含原始回答與來源，方便檢查模型到底說了什麼。
        (folder/'conversation.json').write_text(json.dumps(turns,ensure_ascii=False,indent=2),encoding='utf-8')
    # 環境與參數是實驗證據，不能只留下看起來合理的答案。
    report={'created_at':now(),'seconds':round(time.monotonic()-start,2),'model':llm.metadata(),'index':retriever.manifest,
            'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:version(p) for p in ['mlx','mlx-lm','transformers','torch','numpy','streamlit','requests','beautifulsoup4']}}
    (folder/'environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Markdown 報告只重新排版已生成的內容，不改寫或修正模型答案。
    md=['# 真實地端模型問答紀錄','模型輸出只經 OpenCC 繁體轉換；未人工改寫。原始文字見 conversation.json 的 raw_answer。']
    for i,turn in enumerate(turns,1):
        md.extend([f'## 第 {i} 輪',f'**問題：** {turn["question"]}',turn['answer'],sources_markdown(turn)])
    (folder/'report.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    print(f'結果：{folder}',flush=True)
