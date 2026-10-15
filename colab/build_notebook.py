"""Build download_hamnosys_videos.ipynb from the cell sources below.

Run: python build_notebook.py  (writes the .ipynb next to this file)
"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


md("""
# SignAvatars HamNoSys: tải video RGB về Google Drive

Notebook tải toàn bộ video `video_frontal` trong `data.json` của SignAvatars (5.752 clip),
lưu thành `<mã clip>.mp4` để khớp với tên file `<mã clip>.pkl`.

- **Chạy lại được:** video đã tải sẽ bị bỏ qua. Nếu Colab ngắt, chỉ cần chạy lại từ Cell 1.
- **Log:** `download_log.csv` (mọi lần tải) và `failed.csv` (các clip còn lỗi).
- **Cell 5 (tùy chọn):** kiểm tra số frame video so với `.pkl`.

Chạy lần lượt Cell 1 → 4. Muốn tải lại các clip lỗi thì chạy lại Cell 3 và 4.
""")

code(r'''
# Cell 1: Mount Drive + cấu hình
from google.colab import drive
drive.mount('/content/drive')

import os, glob

# Thư mục lưu video trên Drive (tự tạo nếu chưa có)
OUT_DIR = '/content/drive/MyDrive/Sign Language Datasets/HamNoSys videos'

# Thư mục chứa các file .pkl (bản optimized shape). Để None thì notebook tự tìm.
PKL_DIR = None

NUM_WORKERS = 8   # số luồng tải song song; giảm xuống 4 nếu bị server chặn
MAX_RETRIES = 5

DATA_JSON_URL = ('https://raw.githubusercontent.com/ZhengdiYu/SignAvatars/main/'
                 'datasets/hamnosys2motion/data.json')

if PKL_DIR is None:
    # Tìm thư mục chứa 10010.pkl trong Drive (thư mục giải nén của bản optimized shape)
    hits = glob.glob('/content/drive/MyDrive/**/hamnosys_pkls_cropFalse_shapeTrue/**/10010.pkl',
                     recursive=True)
    PKL_DIR = os.path.dirname(hits[0]) if hits else None
print('PKL_DIR =', PKL_DIR)
if PKL_DIR is None:
    print('Không tìm thấy thư mục .pkl. Nếu thư mục nằm trong "Shared with me", hãy '
          '"Add shortcut to Drive" rồi chạy lại, hoặc gán PKL_DIR thủ công. '
          'Không có PKL_DIR thì vẫn tải đủ 5.752 video.')

os.makedirs(OUT_DIR, exist_ok=True)
print('OUT_DIR =', OUT_DIR)
''')

code(r'''
# Cell 2: Đọc data.json và lập danh sách cần tải
import json, requests

meta_path = os.path.join(OUT_DIR, '_data.json')
if not os.path.exists(meta_path):
    r = requests.get(DATA_JSON_URL, timeout=60)
    r.raise_for_status()
    with open(meta_path, 'wb') as f:
        f.write(r.content)
with open(meta_path, encoding='utf-8') as f:
    data = json.load(f)

def source_of(url):
    if 'slownikpjm' in url: return 'PJM'
    if '/dicta-sign/portal/concepts/gsl/' in url: return 'GSL'
    if '/dicta-sign/portal/concepts/lsf/' in url: return 'LSF'
    return 'DGS'

jobs = {k: v['video_frontal'] for k, v in data.items() if v.get('video_frontal')}

if PKL_DIR:
    pkl_keys = {f[:-4] for f in os.listdir(PKL_DIR) if f.endswith('.pkl')}
    print(f'.pkl: {len(pkl_keys)} | data.json: {len(jobs)} | '
          f'khớp tên: {len(pkl_keys & set(jobs))}')
    missing_url = sorted(pkl_keys - set(jobs))
    if missing_url:
        print('Có .pkl nhưng không có link video:', missing_url[:20])
    jobs = {k: u for k, u in jobs.items() if k in pkl_keys}

from collections import Counter
print('Số video cần tải:', len(jobs), dict(Counter(source_of(u) for u in jobs.values())))
''')

code(r'''
# Cell 3: Tải video (chạy lại được, tự bỏ qua file đã có)
import csv, time, random, shutil, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

LOG_PATH = os.path.join(OUT_DIR, 'download_log.csv')
FAILED_PATH = os.path.join(OUT_DIR, 'failed.csv')
TMP_DIR = '/content/tmp_videos'
os.makedirs(TMP_DIR, exist_ok=True)
log_lock = threading.Lock()
_local = threading.local()

def session():
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
        _local.s.headers['User-Agent'] = 'Mozilla/5.0 (research dataset download)'
    return _local.s

def is_mp4(path):
    with open(path, 'rb') as f:
        head = f.read(12)
    return len(head) == 12 and head[4:8] == b'ftyp'

def log(key, status, nbytes, url, err=''):
    with log_lock:
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['time', 'key', 'status', 'bytes', 'url', 'error'])
            w.writerow([datetime.now().isoformat(timespec='seconds'), key, status, nbytes, url, err])

def download(key, url):
    final = os.path.join(OUT_DIR, f'{key}.mp4')
    tmp = os.path.join(TMP_DIR, f'{key}.mp4')
    err = ''
    for attempt in range(MAX_RETRIES):
        try:
            with session().get(url, stream=True, timeout=(15, 180)) as r:
                if r.status_code == 404:
                    log(key, 'dead_404', 0, url, 'HTTP 404')
                    return key, 'dead_404', 'HTTP 404'
                r.raise_for_status()
                expected = int(r.headers.get('Content-Length', 0))
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            size = os.path.getsize(tmp)
            if expected and size != expected:
                raise IOError(f'incomplete {size}/{expected} bytes')
            if not is_mp4(tmp):
                raise IOError('not an mp4 (missing ftyp box)')
            # Ghi file tạm lên Drive rồi đổi tên, tránh để lại file hỏng khi bị ngắt
            shutil.copyfile(tmp, final + '.part')
            os.replace(final + '.part', final)
            os.remove(tmp)
            log(key, 'ok', size, url)
            return key, 'ok', ''
        except Exception as e:
            err = f'{type(e).__name__}: {e}'[:300]
            time.sleep(min(60, 2 ** attempt + random.random()))
    log(key, 'failed', 0, url, err)
    return key, 'failed', err

existing = {e.name[:-4] for e in os.scandir(OUT_DIR)
            if e.name.endswith('.mp4') and e.stat().st_size > 1000}
todo = {k: u for k, u in jobs.items() if k not in existing}
print(f'Đã có: {len(jobs) - len(todo)} | Cần tải: {len(todo)}')

results = {}
with ThreadPoolExecutor(NUM_WORKERS) as ex:
    futs = [ex.submit(download, k, u) for k, u in todo.items()]
    for fut in tqdm(as_completed(futs), total=len(futs)):
        key, status, err = fut.result()
        results[key] = (status, err)

print(Counter(s for s, _ in results.values()))
''')

code(r'''
# Cell 4: Tổng kết + danh sách lỗi
have = {e.name[:-4] for e in os.scandir(OUT_DIR) if e.name.endswith('.mp4')}
failed = sorted(k for k in jobs if k not in have)
with open(FAILED_PATH, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['key', 'source', 'url', 'last_error'])
    for k in failed:
        w.writerow([k, source_of(jobs[k]), jobs[k], results.get(k, ('', ''))[1]])

print(f'Hoàn tất: {len(jobs) - len(failed)}/{len(jobs)} video')
print('Còn thiếu theo nguồn:', dict(Counter(source_of(jobs[k]) for k in failed)))
print('Danh sách lỗi:', FAILED_PATH)
if failed:
    print('→ Chạy lại Cell 3 để thử tải lại. Clip "dead_404" là link đã chết, tải lại cũng vô ích.')
''')

md("""
## Cell 5 (tùy chọn): kiểm tra số frame video so với `.pkl`

Quy tắc rút ra từ việc kiểm tra 12 clip mẫu (cả 5 nguồn) ngày 2026-10-14:

- **Video ≤ 30 fps** (PJM 24, Dicta-Sign/GLex 25, một số LSF 15): giữ **fps gốc**, frame `.pkl` khớp 1:1.
- **Video 50 fps** (DGS `korpusdict`): `.pkl` khớp với video đã chuyển sang **25 fps bằng `ffmpeg -r 25`**.
- Dòng thứ `i` của `.pkl` ứng với frame `total_valid_index[i]` của video. Một số clip GSL bỏ vài frame
  đầu/cuối (ví dụ `total_valid_index` = 2..71 trên video 74 frame).

Cell này áp quy tắc trên cho mọi clip và ghi `alignment_report.csv`. Cần PKL_DIR.
""")

code(r'''
# Cell 5: Kiểm tra độ khớp frame video ↔ .pkl
import pickle, subprocess
import numpy as np, pandas as pd
import torch  # .pkl chứa tensor torch, cần import để unpickle

def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                          '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_packets',
                          '-of', 'json', path], capture_output=True, text=True).stdout
    s = json.loads(out)['streams'][0]
    num, den = s['avg_frame_rate'].split('/')
    return int(s['width']), int(s['height']), float(num) / float(den), int(s['nb_read_packets'])

def count_at_25(path):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-r', '25', '-f', 'null', '-',
                          '-progress', 'pipe:1'], capture_output=True, text=True).stdout
    return int([l for l in out.splitlines() if l.startswith('frame=')][-1].split('=')[1])

def check(key):
    vid = os.path.join(OUT_DIR, f'{key}.mp4')
    with open(os.path.join(PKL_DIR, f'{key}.pkl'), 'rb') as f:
        d = pickle.load(f)
    n_rows = d['smplx'].shape[0]
    tvi = np.asarray(d['total_valid_index'])
    w, h, fps, n_native = probe(vid)
    n_eff = count_at_25(vid) if fps > 30 else n_native
    if (w, h) != (d['width'], d['height']):
        status = 'size_mismatch'
    elif n_eff == n_rows and tvi.min() == 0:
        status = 'exact'
    elif tvi.max() < n_eff:
        status = 'subset_by_total_valid_index'
    else:
        status = 'MISMATCH'
    return dict(key=key, source=source_of(jobs[key]), width=w, height=h, fps=round(fps, 2),
                video_frames=n_native, effective_frames=n_eff, pkl_rows=n_rows,
                tvi_min=int(tvi.min()), tvi_max=int(tvi.max()), status=status)

keys = sorted(k for k in jobs if os.path.exists(os.path.join(OUT_DIR, f'{k}.mp4')))
rows = []
with ThreadPoolExecutor(4) as ex:
    futs = {ex.submit(check, k): k for k in keys}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        try:
            rows.append(fut.result())
        except Exception as e:
            rows.append(dict(key=futs[fut], status=f'error: {e}'[:200]))

report = pd.DataFrame(rows).sort_values('key')
report.to_csv(os.path.join(OUT_DIR, 'alignment_report.csv'), index=False)
print(report['status'].value_counts())
print(report.groupby(['source', 'status']).size().unstack(fill_value=0))
''')

nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).with_name('download_hamnosys_videos.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote', out)
