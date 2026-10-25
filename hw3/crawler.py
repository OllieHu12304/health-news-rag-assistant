import json
import re
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import BASE_URL, USER_AGENT


def soup_of(html):
    return BeautifulSoup(html, 'html.parser')


# 從首頁找分類網址，不把網站可能改動的分類 UUID 寫死在程式中。
def discover_categories(html, names):
    soup = soup_of(html)
    result = {}
    for link in soup.select('a[href]'):
        name = link.get_text(' ',strip=True)
        # urljoin 可把 /channel/... 這類相對路徑補成完整網址。
        href = urljoin(BASE_URL,link['href'])
        if name in names and '/channel/' in href and urlparse(href).netloc == urlparse(BASE_URL).netloc:
            result[name] = href
    missing = set(names) - result.keys()
    if missing:
        raise ValueError(f'網站分類不存在：{missing}')
    return result


def article_links(html):
    found = []
    # 只讀分類頁的新聞清單，避免把導覽列熱門搜尋或側欄推薦誤當成分類新聞。
    for link in soup_of(html).select('.list-item a[href]'):
        # 去掉查詢參數與頁內錨點，讓同一篇新聞不因追蹤網址不同而重複收錄。
        url = urljoin(BASE_URL,link['href']).split('#')[0].split('?')[0]
        if urlparse(url).netloc == urlparse(BASE_URL).netloc and re.search(r'/article/\d+/?$', url) and url not in found:
            found.append(url)
    return found


def parse_article(html, url, category):
    soup = soup_of(html)
    title_node = soup.select_one('h1')
    title = title_node.get_text(' ',strip=True) if title_node else ''
    # 只從已確認的正文容器取文；找不到時才改讀 JSON-LD 的 articleBody。
    # 不直接使用整頁文字，否則導覽、廣告與頁尾都會混進摘要。
    body = soup.select_one('.article-content-20, .detail-content')
    published = ''
    structured_body = ''
    # JSON-LD 可能是物件、陣列，或把多個物件包在 @graph；統一逐一取出。
    def walk(value):
        if isinstance(value,list):
            for item in value:
                yield from walk(item)
        elif isinstance(value,dict):
            yield value
            if '@graph' in value:
                yield from walk(value['@graph'])
    # 優先使用結構化資料中的 datePublished；這是新聞發布時間，不是爬取時間。
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            for item in walk(json.loads(script.get_text())):
                kind = item.get('@type','')
                if any(t in (kind if isinstance(kind,list) else [kind]) for t in ['NewsArticle','Article','ReportageNewsArticle']):
                    published = item.get('datePublished','') or published
                    structured_body = item.get('articleBody','') or structured_body
                    title = title or item.get('headline','')
        except (ValueError,TypeError):
            continue
    if not published:
        meta = soup.select_one('meta[property="article:published_time"],meta[itemprop="datePublished"]')
        if meta:
            published = meta.get('content','')
    if not published:
        # 結構化日期缺漏時才找日期元件，不從整頁猜日期，以免誤用頁尾時間。
        node = soup.select_one('.detail-date, .article-date, time')
        if node:
            published = node.get('datetime') or node.get_text(' ',strip=True)
    if body:
        for node in body.select('script,style,iframe,.related,.advertisement,.advertise,.social-share'):
            node.decompose()
        paragraphs = [node.get_text(' ',strip=True) for node in body.select('p,h2,h3,li')]
        # dict.fromkeys 在保留段落順序的同時去除重複文字；空段落也排除。
        content = '\n'.join(dict.fromkeys(p for p in paragraphs if p)) or body.get_text('\n',strip=True)
    else:
        content = structured_body.strip()
    # 缺必要欄位就拒絕這篇；不拿空文或現在時間充當有效新聞資料。
    if not title or len(content) < 100 or not published:
        raise ValueError(f'無有效標題、全文或發布時間：{url}（全文 {len(content)} 字，日期 {published!r}）')
    return {'title': title,'url':url,'content':content,'published_at':published,'category':category}


class Crawler:
    def __init__(self, delay=1.5):
        self.delay = max(1.0,delay)
        self.last_request = 0
        # Session 重用 HTTP 連線；User-Agent 表明這是低頻率的教學爬蟲。
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT
        # 遇到限流或暫時性伺服器錯誤時，依退避策略與 Retry-After 重試。
        self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504],respect_retry_after_header=True)))
        self.robots = None

    def get(self, url, check=True):
        if urlparse(url).netloc != urlparse(BASE_URL).netloc:
            raise ValueError('只允許爬取健康醫療網')
        if check and self.robots and not self.robots.can_fetch(USER_AGENT,url):
            raise ValueError(f'robots.txt 不允許讀取：{url}')
        # monotonic 不受系統時鐘調整影響；只補足距上次請求仍欠缺的等待時間。
        time.sleep(max(0,self.delay-(time.monotonic()-self.last_request)))
        # timeout 的兩個值分別是建立連線與等待回應資料的秒數。
        response = self.session.get(url,timeout=(10,40))
        self.last_request = time.monotonic()
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text

    # robots.txt 是用來決定後續可爬路徑的規則，因此讀它本身時不先檢查規則。
    def load_robots(self):
        text = self.get(urljoin(BASE_URL,'robots.txt'),check=False)
        self.robots = RobotFileParser()
        self.robots.parse(text.splitlines())
        return text
