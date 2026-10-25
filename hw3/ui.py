"""工作台的呈現輔助：只處理畫面，不改動檢索或模型資料。"""
import base64
from datetime import datetime
from html import escape
import streamlit as st

# 使用同一個簡單線條圖示；保持 SVG 為固定程式內容，不插入新聞資料。
BOOK_ICON = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke="#176857" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M3 5c4-2 7-1 9 1 2-2 5-3 9-1v14c-4-2-7-1-9 1-2-2-5-3-9-1Z"/><path d="M12 6v14M6 10h3M6 14h3M15 10h3M15 14h3"/></svg>'''


def display_date(value):
    """畫面省略時區尾碼，但下載的原始紀錄仍保留完整 ISO 時間。"""
    try:
        return datetime.fromisoformat(value).strftime('%Y-%m-%d %H:%M')
    except (TypeError, ValueError):
        return str(value)


def brand_header(count):
    # st.html 會清理內嵌 SVG；以靜態圖像資料 URI 呈現，避免依賴外部圖示服務。
    logo=base64.b64encode(BOOK_ICON.encode()).decode()
    image=f'<img src="data:image/svg+xml;base64,{logo}" alt="">'
    st.html(f'''<header class="workbench-header"><div class="brand-group">{image}<h1>健康新聞小幫手</h1><span>新聞研究工作台</span></div><p>本機資料 · {count} 篇新聞</p></header>''')


def question_heading(question):
    # 使用者問題是資料；先跳脫，避免輸入的 HTML 被當成網頁標記執行。
    st.html(f'<div class="question-block"><p>你的問題</p><h3>{escape(question)}</h3></div>')


def source_panel(turn):
    """右欄只呈現目前選取回答的來源，不把其他輪的新聞混進來。"""
    st.subheader('本輪參考資料')
    st.caption('邊讀回答，邊核對新聞。')
    if turn is None:
        st.divider()
        st.write('提出問題後，這裡會顯示參考新聞、發布時間與摘要。')
    else:
        sources = turn['sources']
        status = '本輪檢索結果' if turn['retrieved_this_turn'] else '沿用前文來源 · 本輪未新增檢索'
        st.caption(f'{len(sources)} 篇參考新聞 · {status}')
        st.divider()
        if not sources:
            st.write('本輪沒有新聞來源。')
        for index, source in enumerate(sources, 1):
            # 標題、網址及日期均來自引擎保存的 sources；不是從模型回答中猜測。
            title = escape(source['title'])
            url = escape(source['url'], quote=True)
            date = escape(display_date(source['published_at']))
            st.html(f'''<article class="source-heading"><span class="source-number">{index:02d}</span><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a><p>健康醫療網 · 文章新增時間：{date}</p></article>''')
            with st.expander('查看新聞摘要'):
                st.write(source['summary'])
                if turn['retrieved_this_turn'] and 'score' in source:
                    st.caption(f'向量相似度：{source["score"]:.4f}，不代表回答正確率。')
            st.divider()
    st.caption('這些是提供給模型的參考資料，並非逐句查核結果。請點開新聞原文核對資訊。')
