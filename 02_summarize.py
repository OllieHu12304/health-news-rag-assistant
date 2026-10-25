"""以地端模型摘要所有尚未處理的新聞；每篇完成即保存，可中斷續跑。"""
import argparse
import time
import json
from hw3.config import ROOT
from hw3.db import now
from hw3.db import articles,save_summary
from hw3.llm import LocalLLM

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--redo',action='store_true',help='先備份既有摘要，再重新生成')
    args=parser.parse_args()
    # 預設只處理還沒有摘要的文章，因此中斷後可直接用相同命令續跑。
    rows=[row for row in articles() if args.redo or not row['summary']]
    # 明確要求重做時，先備份現有摘要及其生成紀錄，再開始取代資料庫內容。
    if args.redo:
        backup=ROOT/'results'/('summary_backup_'+now().replace(':','').replace('+','_')+'.json')
        backup.write_text(json.dumps([{k:r[k] for k in ('id','title','url','summary','summary_meta')} for r in rows],ensure_ascii=False,indent=2),encoding='utf-8')
    if args.limit is not None:
        if args.limit<1: parser.error('--limit 必須大於 0')
        rows=rows[:args.limit]
    if not rows: print('沒有待摘要文章；請先執行 01_crawl.py 或檢查資料庫。')
    else:
        # 模型只載入一次，多篇文章共用，避免每篇都支付載入權重的成本。
        llm=LocalLLM()
        for i,row in enumerate(rows,1):
            start=time.monotonic()
            summary,meta=llm.summarize(row)
            meta['seconds']=round(time.monotonic()-start,2)
            # 每篇生成完就提交資料庫，不需要等整批跑完才保存。
            save_summary(row['id'],summary,meta)
            print(f'[{i}/{len(rows)}] {row["title"]} ({meta["seconds"]} 秒)\n{summary}\n',flush=True)
