"""從新聞分類慢速爬取文章，保存原始 HTML 方便檢查解析結果。"""
import argparse
import json
from hw3.config import BASE_URL, DATA, ROOT
from hw3.crawler import Crawler, discover_categories, article_links, parse_article, soup_of
from urllib.parse import urljoin
from hw3.db import upsert, now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-category',type=int,default=15)
    parser.add_argument('--categories',nargs='+',default=['健康保健','營養飲食'])
    args=parser.parse_args()
    if args.per_category < 1:
        parser.error('--per-category 必須大於 0')
    crawler=Crawler()
    DATA.mkdir(exist_ok=True)
    # 先讀網站爬蟲規則，再由首頁找出指定分類的連結。
    (DATA/'robots.txt').write_text(crawler.load_robots())
    categories=discover_categories(crawler.get(BASE_URL),args.categories)
    raw=DATA/'raw';raw.mkdir(exist_ok=True)
    report={'started_at':now(),'categories':categories,'articles':[],'errors':[]}
    # 跨分類共用已看過的網址；同一篇新聞不會因同時出現在兩類而重複計數。
    seen=set()
    for category,url in categories.items():
        count=0
        page_url=url
        # 每類最多巡覽 10 頁，避免分類文章不足時無止境往下爬。
        for page in range(1, 11):
            listing=crawler.get(page_url)
            (raw/f'{category}_{page}.html').write_text(listing,encoding='utf-8')
            for link in article_links(listing):
                if link in seen:
                    continue
                seen.add(link)
                try:
                    html=crawler.get(link)
                    article=parse_article(html,link,category)
                    # upsert 決定新增／更新／略過；changed 記錄本次是否改動資料庫。
                    changed=upsert(article)
                    # HTML 保存於本機，之後網站結構改動時可用來重現解析問題。
                    (raw/f"{link.rstrip('/').split('/')[-1]}.html").write_text(html,encoding='utf-8')
                    report['articles'].append({'url':link,'title':article['title'],'published_at':article['published_at'],'characters':len(article['content']),'category':category,'changed':changed})
                    count+=1
                    print(f'[{category} {count}/{args.per_category}] {article["title"]}',flush=True)
                    if count>=args.per_category: break
                except (ValueError, OSError) as exc:
                    report['errors'].append({'url':link,'error':str(exc)})
                    print(f'跳過：{exc}',flush=True)
            # 使用網站實際提供的下一頁連結，而不是自行猜測分頁網址。
            next_link=soup_of(listing).select_one('a[rel="next"][href]')
            if count>=args.per_category or not next_link: break
            page_url=urljoin(BASE_URL,next_link['href'])
        if count<args.per_category:
            report['errors'].append({'category':category,'error':f'分類頁只有 {count} 篇有效不重複文章；未湊造資料'})
    # 報告保留來源、時間與失敗原因；部分文章失敗不等於整批成果都要丟棄。
    report['finished_at']=now()
    (ROOT/'results'/'crawl_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['articles']:
        raise SystemExit('未取得有效新聞，請檢查 results/crawl_report.json 與網站結構')


if __name__=='__main__': main()
