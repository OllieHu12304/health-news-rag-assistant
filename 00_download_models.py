"""重用既有未微調 MLX 模型，或下載固定 revision；不載入笑話 adapter。"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from hw3.config import ROOT, MODEL_PATH, MODEL_ID, EMBED_PATH, EMBED_ID


def obtain(repo_id, path, reuse=None):
    # 目錄已存在就略過，避免每次啟動都重新下載大型權重。
    if path.exists():
        print(f'模型已存在：{path}',flush=True)
        return
    # 優先重用作業二已下載的原始模型；符號連結只指向原目錄，不複製權重。
    # 連結本身不會限制寫入權限，本程式只把它作為推論讀取來源。
    if reuse and (reuse/'config.json').exists():
        path.parent.mkdir(exist_ok=True,parents=True)
        path.symlink_to(reuse.resolve(),target_is_directory=True)
        print(f'重用唯讀基礎模型：{reuse}',flush=True)
        return
    # revision 是固定的模型版本指紋；不用隨時間改變的 main，方便重現實驗。
    revision=json.loads((ROOT/'model_sources.json').read_text())[repo_id]
    print(f'下載 {repo_id}@{revision}',flush=True)
    # 只下載載入模型所需的設定、tokenizer 與 safetensors 等檔案。
    patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md']
    snapshot_download(repo_id,revision=revision,local_dir=path,allow_patterns=patterns)
    # 保存實際下載的版本，建立索引時會把 embedding 版本一起寫入 manifest。
    (path/'hw3_model_source.json').write_text(json.dumps({'repo_id':repo_id,'revision':revision},indent=2))


if __name__=='__main__':
    obtain(MODEL_ID,MODEL_PATH,ROOT.parent/'nlp_hw2/models/Qwen2.5-3B-Instruct-4bit')
    obtain(EMBED_ID,EMBED_PATH)
