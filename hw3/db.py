# SQLite 是新聞與摘要的資料來源；向量檔案另外保存，但必須能對回這裡的 id。
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from .config import DB_PATH


def now():
    # 明確使用臺北時區，讓爬取、摘要與示範紀錄的時間格式一致。
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')


@contextmanager
def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    # 讓查詢結果可用 row["title"] 取值，比記憶欄位的數字位置容易閱讀。
    con.row_factory = sqlite3.Row
    try:
        con.execute('''CREATE TABLE IF NOT EXISTS articles (
          id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
          url TEXT NOT NULL UNIQUE, content TEXT NOT NULL, summary TEXT,
          published_at TEXT NOT NULL, crawled_at TEXT NOT NULL,
          category TEXT NOT NULL, content_hash TEXT NOT NULL,
          summary_meta TEXT, summarized_at TEXT)''')
        # with connect() 的呼叫端在這裡使用連線；正常結束才提交交易。
        yield con
        con.commit()
    except BaseException:
        # 任何例外（包含使用者中斷）都回復尚未提交的修改，再把錯誤傳回呼叫端。
        con.rollback()
        raise
    finally:
        con.close()


def upsert(article, path=DB_PATH):
    # 全文雜湊是內容指紋：網址相同，也能判斷網站文章是否已被修改。
    digest = hashlib.sha256(article['content'].encode()).hexdigest()
    with connect(path) as con:
        previous = con.execute('SELECT content_hash, title FROM articles WHERE url=?', (article['url'],)).fetchone()
        if previous and previous['content_hash'] == digest and previous['title'] == article['title']:
            return False
        # upsert = 有同網址就更新，沒有就新增；標題或全文變動後，舊摘要必須失效。
        # 問號是 SQL 參數占位符，實際文字由後面的 tuple 傳入，不直接拼成 SQL。
        con.execute('''INSERT INTO articles(title,url,content,published_at,crawled_at,category,content_hash)
          VALUES(?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET
          title=excluded.title, content=excluded.content, published_at=excluded.published_at,
          crawled_at=excluded.crawled_at, category=excluded.category, content_hash=excluded.content_hash,
          summary=NULL,summary_meta=NULL,summarized_at=NULL''',
          (article['title'],article['url'],article['content'],article['published_at'],now(),article['category'],digest))
    return True


def articles(path=DB_PATH, summarized=False):
    with connect(path) as con:
        query = 'SELECT * FROM articles'
        if summarized:
            query += " WHERE summary IS NOT NULL AND length(trim(summary)) > 0"
        # 固定依 id 排序，讓向量矩陣的每一列與新聞清單順序保持一致。
        return [dict(row) for row in con.execute(query + ' ORDER BY id')]


# 每篇摘要單獨寫入，長批次工作中斷時，已完成的文章仍會保留。
def save_summary(article_id, summary, metadata, path=DB_PATH):
    if not summary.strip():
        raise ValueError('模型回傳空摘要，未寫入資料庫')
    with connect(path) as con:
        con.execute('UPDATE articles SET summary=?,summary_meta=?,summarized_at=? WHERE id=?',
                    (summary.strip(),json.dumps(metadata,ensure_ascii=False),now(),article_id))


def corpus_hash(rows):
    # 只取會影響檢索與引用的欄位，並固定 JSON 鍵順序，產生可重現的語料指紋。
    # 之後載入索引時會重算它，避免用舊向量回答更新後的新聞。
    payload = [{key: row[key] for key in ('id','title','url','summary','published_at','content_hash')} for row in rows]
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
