# 共用設定集中在這裡，讓命令列腳本與網頁使用相同的模型、資料庫與索引。
import os
from pathlib import Path

# 依程式檔案位置定位專案，而不是依賴目前終端機所在目錄。
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DB_PATH = DATA / 'news.sqlite3'
INDEX_PATH = DATA / 'index.npz'
MANIFEST_PATH = DATA / 'index_manifest.json'
# MODEL_ID 是 Hugging Face 名稱；MODEL_PATH 是實際讀取權重的本機路徑。
MODEL_ID = 'mlx-community/Qwen2.5-3B-Instruct-4bit'
# 環境變數可覆寫預設路徑，避免為了換模型位置而修改程式。
MODEL_PATH = Path(os.getenv('HW3_MODEL_PATH', str(ROOT / 'models' / 'Qwen2.5-3B-Instruct-4bit'))).expanduser()
# 生成模型負責摘要／回答；embedding 模型只負責把文字轉成檢索向量。
EMBED_ID = 'intfloat/multilingual-e5-small'
EMBED_PATH = ROOT / 'models' / 'multilingual-e5-small'
BASE_URL = 'https://www.healthnews.com.tw/'
USER_AGENT = 'NLPHomework3/0.1 (educational news retrieval; low-rate crawler)'
