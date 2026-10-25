import unittest
from hw3.crawler import parse_article, article_links

class CrawlerTests(unittest.TestCase):
    def test_current_body_container(self):
        html='<h1>新聞標題</h1><script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-10-21T09:00:00+08:00"}</script><div class="article-content-20"><p>'+('真實正文。'*40)+'</p></div><footer>不應納入</footer>'
        row=parse_article(html,'https://www.healthnews.com.tw/article/1','健康保健')
        self.assertIn('真實正文',row['content'])
        self.assertNotIn('不應納入',row['content'])
        self.assertEqual(row['published_at'],'2026-10-21T09:00:00+08:00')
    def test_missing_body_rejected(self):
        with self.assertRaises(ValueError): parse_article('<h1>新聞</h1>','x','y')
    def test_category_only(self):
        html='<nav><a href="/article/999">熱搜</a></nav><li class="list-item"><a href="/article/123">新聞</a></li>'
        self.assertEqual(article_links(html),['https://www.healthnews.com.tw/article/123'])
