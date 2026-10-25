"""啟動：uv run streamlit run app.py --server.port 8503。B 新聞研究工作台。"""
import json
import streamlit as st
from hw3.config import ROOT
from hw3.db import articles
from hw3.retrieval import load_index
from hw3.chat import ChatEngine, SUMMARY_PROMPT, sources_markdown
from hw3.ui import brand_header, question_heading, source_panel, display_date

st.set_page_config(page_title='健康新聞小幫手',page_icon=None,layout='wide')
# 版面樣式獨立存放；Python 仍負責原生元件與狀態，CSS 不操作對話資料。
st.html(f'<style>{(ROOT / "assets/workbench.css").read_text(encoding="utf-8")}</style>')

# turns 保存完整對話，selected_turn 只決定畫面正在查看哪一輪。
# 切換舊問題不會刪除歷史；下一次送入模型時仍使用完整 turns。
if 'turns' not in st.session_state:
    st.session_state.turns=[]
if 'selected_turn' not in st.session_state:
    st.session_state.selected_turn=None
rows=articles()
ready=False
try:
    _,_,manifest=load_index()
    ready=True
except (FileNotFoundError,ValueError,KeyError) as exc:
    status_error=str(exc)

# 沿用同一份地端引擎；首頁與問題切換不會觸發模型載入或生成。
@st.cache_resource(show_spinner=False)
def resources():
    from hw3.llm import LocalLLM
    from hw3.retrieval import Retriever
    return ChatEngine(LocalLLM(),Retriever())

with st.sidebar:
    st.header('本次對話')
    if st.button('開始新對話',icon=':material/add:',use_container_width=True):
        st.session_state.turns=[]
        st.session_state.selected_turn=None
        st.rerun()
    if st.session_state.turns:
        st.caption('點選問題，查看該輪回答與參考新聞。')
        for index,turn in enumerate(st.session_state.turns):
            title='本次對話重點' if turn['mode']=='summary' else turn['question']
            # 限制按鈕文字長度，完整問題仍會出現在主區與提示文字中。
            label=f'第 {index+1} 輪 · {title[:22]}'+('…' if len(title)>22 else '')
            selected=st.session_state.selected_turn==index
            if st.button(label,key=f'history_{index}',type='primary' if selected else 'secondary',help=turn['question'],use_container_width=True):
                st.session_state.selected_turn=index
                st.rerun()
    else:
        st.caption('你的問題會依序保存在這裡。')
    st.divider()
    summary=st.button('整理本次對話重點',icon=':material/summarize:',disabled=not st.session_state.turns or not ready,use_container_width=True)
    if st.button('清除對話',icon=':material/delete_outline:',disabled=not st.session_state.turns,use_container_width=True):
        st.session_state.turns=[]
        st.session_state.selected_turn=None
        st.rerun()
    if st.session_state.turns:
        st.download_button('下載對話紀錄',icon=':material/download:',data=json.dumps(st.session_state.turns,ensure_ascii=False,indent=2),file_name='health_news_chat.json',mime='application/json',use_container_width=True)
    st.divider()
    st.subheader('新聞資料')
    st.write(f'{len(rows)} 篇新聞 · {sum(bool(r["summary"]) for r in rows)} 篇已摘要')
    st.caption('健康保健 · 營養飲食')
    if ready:
        st.caption(f'索引建立：{display_date(manifest["created_at"])}')
    with st.expander('查看新聞清單'):
        for row in rows:
            st.link_button(row['title'],row['url'])
    st.caption('模型可能誤讀新聞，請點開原文核對。健康資訊不取代醫師診斷。')

# 第一次載入既有 session 或清除後，避免選取超出目前歷史範圍。
turns=st.session_state.turns
selected=st.session_state.selected_turn
if turns and (selected is None or not 0<=selected<len(turns)):
    selected=len(turns)-1
    st.session_state.selected_turn=selected
current=turns[selected] if turns else None
brand_header(len(rows))
question=None
suggested=None

# Streamlit 原生 columns 承擔兩個工作區；CSS 在850px以下將其堆疊。
with st.container(key='workbench'):
    chat_column,evidence_column=st.columns([2.35,1],gap='large')
    with chat_column:
        with st.container(key='conversation_panel'):
            st.subheader('新聞問答' if not current or current['mode']!='summary' else '本次對話重點')
            if current:
                status='本輪有檢索新聞' if current['retrieved_this_turn'] else '本輪未新增檢索'
                st.caption(f'第 {selected+1} 輪 · {status}')
            else:
                st.caption('根據收錄新聞整理回答，並列出參考來源。')
            if not ready:
                st.info(status_error)
                st.write('請在作業三資料夾依序完成：')
                st.code('uv run python 00_download_models.py\nuv run python 01_crawl.py\nuv run python 02_summarize.py\nuv run python 03_build_index.py',language='bash')
                st.button('重新檢查資料',on_click=st.rerun)
            elif current is None:
                with st.container(key='empty_state'):
                    st.markdown('### 從一個問題開始，找到新聞裡的線索。')
                    st.write('說說你想了解的主題。回答會附上參考新聞，讓你繼續閱讀、核對資訊，也能整理整段對話。')
                    if st.button('吃飯速度與減重有什麼關係？',use_container_width=True):
                        suggested='吃飯速度與減重有什麼關係？上班族可以怎麼調整？'
                    if st.button('低 FODMAP 飲食是什麼？',use_container_width=True):
                        suggested='低 FODMAP 飲食的目的與執行方式是什麼？'
            if current is not None:
                question_heading(current['question'])
                with st.chat_message('assistant',avatar=':material/menu_book:'):
                    st.caption('新聞小幫手 · 地端模型回答')
                    st.markdown(current['answer'])
                # 每輪回答仍附引用；右欄另外提供摘要，讓桌面使用者能同時核對。
                with st.container(key='inline_citations'):
                    with st.expander(f'參考新聞與文章新增時間 · {len(current["sources"])} 篇'):
                        st.markdown(sources_markdown(current))
            with st.container(key='composer'):
                # 放在欄位內的 chat_input 會就地顯示，不會固定於整頁底部蓋住來源。
                retrieve=st.checkbox('本輪檢索新聞',value=True,key='retrieve_news',help='關閉後只使用完整對話中已取得的新聞。整理重點不新增檢索。')
                question=st.chat_input('輸入你想了解的健康新聞問題',disabled=not ready,max_chars=1500,submit_mode='disable')
                st.caption('可接著追問，或到左側整理整段對話。')
                if current is not None and selected!=len(turns)-1:
                    st.caption('目前查看的是較早回答；新問題仍會接續最新的完整對話。')
    with evidence_column:
        with st.container(key='evidence_panel'):
            source_panel(current)

if suggested:
    question=suggested
if summary:
    question=SUMMARY_PROMPT
if question:
    # 顯示新問題的當下才真正生成；成功後選取新一輪，左右區域便會同步更新。
    with chat_column:
        try:
            with st.spinner('正在讀取新聞並以地端模型整理回答，首次使用需要載入模型…'):
                turn=resources().answer(question,turns,retrieve=retrieve,summarize=summary)
            st.session_state.turns.append(turn)
            st.session_state.selected_turn=len(st.session_state.turns)-1
            st.rerun()
        except (FileNotFoundError,ValueError,RuntimeError,OSError) as exc:
            st.error(f'暫時無法回答：{exc}')
            st.caption('請確認資料、模型與索引已建立，或清除過長的對話後再試。')
