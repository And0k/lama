# %% [markdown]
# # LaMa-inpainting для 2D-разрезов T океана
# **Балтийское море | Синтетические данные | Colab GPU**
#
# Запуск: Runtime → Change runtime type → GPU (T4)
# Весь пайплайн занимает ~2–3ч на T4 (120 эпох, 400 сэмплов).

# %% [code] — Setup
# !pip install torch torchvision scipy matplotlib numpy -q

import sys, os
# если запускаем в Colab — монтируем Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/lama_ocean_results'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Colab + Drive OK")
except ImportError:
    SAVE_DIR = './output'
    print("Local mode")

# %% [code] — Импорт и конфигурация
# Скопируйте сюда весь train.py ИЛИ выполните:
# !wget -q https://raw.githubusercontent.com/YOUR_FORK/lama_ocean/main/train.py

# Быстрый режим для отладки (уменьшите перед полным запуском):
FAST_DEBUG = False   # True → 10 эпох, 40 сэмплов, видно за 5 мин

if FAST_DEBUG:
    CFG.update(dict(
        EPOCHS=10, N_SAMPLES_TRAIN=40, N_SAMPLES_VAL=16,
        BATCH_SIZE=4, VIZ_EVERY=5,
    ))

# %% [code] — Обучение
# Запустите train.py как модуль:
# exec(open('train.py').read())
# ИЛИ если установлен как пакет:
# from train import main; main()

# %% [code] — Просмотр результатов
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

out = Path(SAVE_DIR) if SAVE_DIR else Path('./output')

for img_path in sorted(out.glob('comparison_ep*.png')):
    img = mpimg.imread(img_path)
    plt.figure(figsize=(18, 4))
    plt.imshow(img); plt.axis('off')
    plt.title(img_path.stem); plt.show()

# %% [code] — Таблица метрик
import json, pandas as pd

with open(out / 'eval_table.json') as f:
    tbl = json.load(f)

df = pd.DataFrame(tbl)
df['epoch'] = df['epoch'].replace(-1, 'BEST')
df = df.set_index('epoch')
df['ΔRMSE'] = df['oi_rmse'] - df['lama_rmse']   # > 0 → LaMa лучше OI
df['ΔPSNR'] = df['lama_psnr'] - df['oi_psnr']   # > 0 → LaMa лучше OI
print(df.to_string())

# %% [code] — Кривые потерь
img = mpimg.imread(out / 'loss_curves.png')
plt.figure(figsize=(12,4)); plt.imshow(img); plt.axis('off'); plt.show()
