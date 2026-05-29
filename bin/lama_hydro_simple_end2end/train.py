"""
LaMa-генератор для inpainting 2D-разрезов T/S океана.

Вход: (C, H, W) где C = [field, mask_ctd, mask_cmems, bathymetry]
    - field      : частично заполненное поле (CTD-линии + CMEMS-точки, остальное=0)
    - mask_ctd   : 1 там где CTD-профиль (вертикальная линия)
    - mask_cmems : 1 там где CMEMS-точки (разреженная сетка с ростом шага по z)
    - bathymetry : нормированная батиметрия (дно = 1)
Выход: (1, H, W) — плотное поле T

Физические лоссы (§1.1):
    L_stab : штраф за ∂ρ/∂z < 0  (гидростатическая неустойчивость)
    L_bbl  : штраф за |∂T/∂z| у дна  (гомогенизация BBL)
    L_grad : TV-регуляризация  (подавление нефизичных артефактов)

GAN отключён: PatchGAN-дискриминатор не нужен для гладких физических полей,
удваивает время обучения и нестабилен при малых данных (§2.2).
"""

import math, time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── dirs ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT  = ROOT / "output";  OUT.mkdir(exist_ok=True)
CKPT = ROOT / "checkpoints"; CKPT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # домен
    NX = 64,          # точки по горизонтали (расстояние)
    NZ = 64,          # точки по вертикали   (глубина)
    N_CTD  = 5,       # CTD-профилей на разрез
    N_CMEMS_X = 8,    # CMEMS-колонок по x
    # синтетические данные
    N_SAMPLES_TRAIN = 400,
    N_SAMPLES_VAL   = 80,
    NOISE_CTD   = 0.05,   # σ шума CTD
    NOISE_CMEMS = 0.10,   # σ шума CMEMS
    # обучение
    EPOCHS     = 120,
    BATCH_SIZE = 4,
    LR         = 2e-4,
    LR_PATIENCE= 15,
    ES_PATIENCE= 30,      # early stopping
    # веса лоссов
    W_DATA  = 1.0,
    W_STAB  = 0.3,
    W_BBL   = 0.2,
    W_GRAD  = 0.1,
    # архитектура LaMa-lite
    BASE_CH  = 32,
    N_BLOCKS = 6,
    # устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    VIZ_EVERY = 20,
)

print(f"Device: {CFG['DEVICE']} | NX={CFG['NX']} NZ={CFG['NZ']}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. СИНТЕТИЧЕСКИЕ ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════
def _linspace(n, device="cpu"):
    return torch.linspace(0, 1, n, device=device)


def synthetic_T_field(nx, nz, rng: np.random.Generator):
    """
    Реалистичный 2D-разрез T: термоклин + горизонтальный градиент + вихрь.
    Возвращает ndarray (nz, nx) нормированное в [0,1].
    """
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)   # 0=поверхность, 1=дно
    XX, ZZ = np.meshgrid(x, z)

    # термоклин: сигмоида по z
    z_thermo = rng.uniform(0.15, 0.40)
    sharpness = rng.uniform(8, 20)
    T = 1.0 / (1.0 + np.exp(sharpness * (ZZ - z_thermo)))

    # горизонтальный наклон
    T += rng.uniform(-0.15, 0.15) * XX

    # случайный гауссов вихрь
    cx, cz = rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.5)
    rx, rz = rng.uniform(0.1, 0.25), rng.uniform(0.05, 0.15)
    amp    = rng.uniform(-0.25, 0.25)
    T     += amp * np.exp(-((XX-cx)**2/(2*rx**2) + (ZZ-cz)**2/(2*rz**2)))

    T = (T - T.min()) / (T.max() - T.min() + 1e-8)
    return T.astype(np.float32)


def sloping_bathymetry(nx, rng: np.random.Generator):
    """Нормированная глубина дна [0,1] для каждого x; монотонно растёт."""
    base   = np.linspace(rng.uniform(0.5, 0.7), rng.uniform(0.75, 0.95), nx)
    noise  = rng.normal(0, 0.02, nx).cumsum() * 0.1
    bathy  = np.clip(base + noise, 0.4, 0.99)
    return bathy.astype(np.float32)


def cmems_z_indices(nz, n_cols=8):
    """
    Квази-регулярная сетка по z с нарастающим (неравномерным) интервалом —
    имитирует стандартные уровни CMEMS (плотнее у поверхности).
    Возвращает массив индексов.
    """
    # логарифмическое сгущение к поверхности
    raw  = np.logspace(0, np.log10(nz), n_cols, endpoint=False)
    idxs = np.unique(np.clip(raw.astype(int), 0, nz-1))
    return idxs


def make_sample(nx, nz, n_ctd, n_cmems_x, noise_ctd, noise_cmems,
                rng: np.random.Generator):
    """
    Возвращает dict:
        field    (4, nz, nx) float32 — вход сети
        target   (1, nz, nx) float32 — полное поле
        bathy_row (nx,)      float32 — индексы дна (в долях nz)
    """
    T     = synthetic_T_field(nx, nz, rng)       # (nz, nx)
    bathy = sloping_bathymetry(nx, rng)          # (nx,) в [0,1]

    # маска дна: пиксели ниже дна = True
    z_norm = np.linspace(0, 1, nz)[:, None]     # (nz,1)
    below  = z_norm > bathy[None, :]             # (nz, nx) bool

    T[below] = 0.0   # за пределами домена — нули

    # --- CTD-профили: случайные x-позиции, вертикальная линия ---------------
    ctd_xs   = np.sort(rng.choice(nx, n_ctd, replace=False))
    mask_ctd = np.zeros((nz, nx), np.float32)
    field    = np.zeros((nz, nx), np.float32)

    for xi in ctd_xs:
        depth_col = int(bathy[xi] * nz)
        mask_ctd[:depth_col, xi] = 1.0
        field[:depth_col, xi]    = (T[:depth_col, xi]
                                    + rng.normal(0, noise_ctd, depth_col).astype(np.float32))

    # --- CMEMS-точки: разреженная квази-регулярная сетка --------------------
    cmems_zs = cmems_z_indices(nz, n_cmems_x)
    cmems_xs = np.linspace(0, nx-1, n_cmems_x, dtype=int)
    mask_cmems = np.zeros((nz, nx), np.float32)

    for zi in cmems_zs:
        for xi in cmems_xs:
            if z_norm[zi, 0] < bathy[xi]:
                mask_cmems[zi, xi] = 1.0
                field[zi, xi] = (T[zi, xi]
                                 + np.float32(rng.normal(0, noise_cmems)) * (1 - mask_ctd[zi, xi]))

    # батиметрия как канал (нормированная глубина дна, broadcast по z)
    bathy_ch = np.tile(bathy[None, :], (nz, 1)).astype(np.float32)

    inp = np.stack([field, mask_ctd, mask_cmems, bathy_ch], axis=0)  # (4, nz, nx)
    tgt = T[None]                                                      # (1, nz, nx)

    return dict(input=inp, target=tgt, bathy=bathy, below=below)


class OceanDataset(Dataset):
    def __init__(self, n, cfg, seed=0):
        self.rng = np.random.default_rng(seed)
        self.samples = [make_sample(
            cfg['NX'], cfg['NZ'], cfg['N_CTD'], cfg['N_CMEMS_X'],
            cfg['NOISE_CTD'], cfg['NOISE_CMEMS'], self.rng
        ) for _ in range(n)]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return (torch.from_numpy(s['input']),
                torch.from_numpy(s['target']),
                torch.from_numpy(s['bathy']),
                torch.from_numpy(s['below'].astype(np.float32)))


# ══════════════════════════════════════════════════════════════════════════════
# 3. АРХИТЕКТУРА LaMa-LITE (генератор, без дискриминатора)
# ══════════════════════════════════════════════════════════════════════════════
class FourierUnit(nn.Module):
    """
    Fast Fourier Convolution unit: глобальное поле восприятия за O(N log N).
    Операции в частотной области → умножение ≡ глобальная свёртка (§2.2).
    """
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch * 2, 1)   # на действ. и мним. части
        self.bn   = nn.BatchNorm2d(ch * 2)

    def forward(self, x):
        b, c, h, w = x.shape
        f  = torch.fft.rfft2(x, norm="ortho")           # (b, c, h, w//2+1) complex
        fr = torch.cat([f.real, f.imag], dim=1)         # (b, 2c, h, w//2+1)
        fr = F.relu(self.bn(self.conv(fr)))
        c2 = fr.shape[1] // 2
        f  = torch.complex(fr[:, :c2], fr[:, c2:])
        return torch.fft.irfft2(f, s=(h, w), norm="ortho")


class FFCResBlock(nn.Module):
    """Residual block: локальная 3×3 свёртка + FourierUnit → конкатенация."""
    def __init__(self, ch):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.global_branch = FourierUnit(ch)
        self.fuse = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, x):
        loc = self.local_branch(x)
        glb = self.global_branch(x)
        return x + self.fuse(torch.cat([loc, glb], 1))


class LamaGenerator(nn.Module):
    """
    Энкодер → N × FFCResBlock → декодер.
    Вход: (B, 4, H, W); выход: (B, 1, H, W) ∈ [0,1].
    Параметров: ~0.5M при BASE_CH=32, N_BLOCKS=6 — умещается в Colab.
    """
    def __init__(self, in_ch=4, base_ch=32, n_blocks=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3), nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.blocks = nn.Sequential(*[FFCResBlock(base_ch*2) for _ in range(n_blocks)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch, 1, 7, padding=3), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.blocks(self.enc(x)))


# ══════════════════════════════════════════════════════════════════════════════
# 4. ФУНКЦИИ ПОТЕРЬ
# ══════════════════════════════════════════════════════════════════════════════
def loss_data(pred, target, mask_obs, below):
    """
    MSE только в наблюдаемых точках (CTD+CMEMS) + мягкий штраф в области домена.
    mask_obs : (B,1,H,W) — объединённая маска наблюдений
    below    : (B,1,H,W) — за пределами домена (дно)
    """
    domain = 1.0 - below
    l_obs  = F.mse_loss(pred * mask_obs,  target * mask_obs)
    l_dom  = F.mse_loss(pred * domain,    target * domain) * 0.1
    return l_obs + l_dom


def _density(T_norm):
    """
    Линеаризованная плотность ρ ≈ ρ₀(1 - α·T̃), T̃ ∈ [0,1].
    Монотонная функция T → достаточно для знака ∂ρ/∂z.
    """
    alpha = 0.25
    return 1.0 - alpha * T_norm


def loss_stability(pred, below):
    """
    L_stab: штраф за ∂ρ/∂z < 0 (гидростатическая неустойчивость).
    ∂ρ/∂z < 0 ⟺ ∂T/∂z > 0 (при линеаризованном уравнении состояния).
    Центральная разность по оси z (dim=2).
    """
    domain = 1.0 - below
    dT_dz  = pred[:, :, 2:, :] - pred[:, :, :-2, :]   # (B,1,H-2,W)
    mask   = domain[:, :, 2:, :] * domain[:, :, :-2, :]
    unstable = F.relu(dT_dz)                            # только dT/dz > 0
    return (unstable * mask).mean()


def loss_bbl(pred, bathy, nz):
    """
    L_bbl: вертикальный градиент T у дна → 0 (гомогенизация BBL).
    bathy: (B, W) ∈ [0,1] → переводим в индексы.
    """
    b, _, h, w = pred.shape
    total = torch.tensor(0.0, device=pred.device)
    n     = 0
    for bi in range(b):
        for xi in range(w):
            zi = int(bathy[bi, xi].item() * nz) - 2
            if zi > 1:
                dT = pred[bi, 0, zi, xi] - pred[bi, 0, zi-1, xi]
                total += dT ** 2
                n     += 1
    return total / max(n, 1)


def loss_grad_tv(pred, below):
    """
    L_grad: изотропная TV-регуляризация — подавляет нефизичные артефакты,
    не сглаживая реальные высокие градиенты (термоклин).
    """
    domain = 1.0 - below
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) * domain[:, :, 1:, :]
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) * domain[:, :, :, 1:]
    return (dy.abs().mean() + dx.abs().mean()) * 0.5


def compute_loss(pred, target, inp, bathy, below, cfg):
    mask_obs = torch.clamp(inp[:, 1:2] + inp[:, 2:3], 0, 1)
    below_4  = below.unsqueeze(1)

    l_d = loss_data(pred, target, mask_obs, below_4)
    l_s = loss_stability(pred, below_4)
    l_b = loss_bbl(pred, bathy, cfg['NZ'])
    l_g = loss_grad_tv(pred, below_4)

    total = (cfg['W_DATA'] * l_d + cfg['W_STAB'] * l_s
             + cfg['W_BBL'] * l_b + cfg['W_GRAD'] * l_g)
    return total, dict(data=l_d.item(), stab=l_s.item(),
                       bbl=l_b.item(), grad=l_g.item(), total=total.item())


# ══════════════════════════════════════════════════════════════════════════════
# 5. OI-БАЗЕЛАЙН
# ══════════════════════════════════════════════════════════════════════════════
def optimal_interpolation(inp_np, nx, nz):
    """
    scipy.griddata (метод cubic) по всем известным точкам (CTD+CMEMS).
    Возвращает ndarray (nz, nx).
    """
    field    = inp_np[0]
    mask_all = np.clip(inp_np[1] + inp_np[2], 0, 1)

    pts = np.argwhere(mask_all > 0.5)         # (K, 2): [zi, xi]
    if len(pts) < 4:
        return np.zeros((nz, nx), np.float32)

    vals   = field[pts[:, 0], pts[:, 1]]
    zi_all = np.arange(nz)
    xi_all = np.arange(nx)
    ZI, XI = np.meshgrid(zi_all, xi_all, indexing='ij')
    grid   = np.stack([ZI.ravel(), XI.ravel()], axis=1)

    interp = griddata(pts, vals, grid, method='cubic', fill_value=0.0)
    return interp.reshape(nz, nx).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(pred, target, below):
    """
    pred, target, below: ndarray (nz, nx).
    Метрики считаются только в области домена (не ниже дна) и вне наблюдений.
    """
    domain = ~below
    if domain.sum() == 0:
        return dict(mse=np.nan, mae=np.nan, psnr=np.nan, rmse=np.nan)

    p = pred[domain]; t = target[domain]
    mse  = float(np.mean((p - t)**2))
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))
    psnr = float(20 * np.log10(1.0 / (rmse + 1e-8))) if rmse > 0 else np.inf
    return dict(mse=mse, mae=mae, rmse=rmse, psnr=psnr)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison(sample, pred_lama, pred_oi, metrics_lama, metrics_oi,
                    epoch, save_path):
    """
    6-панельный график:
    [True | LaMa-pred | OI-pred | LaMa-err | OI-err | маска наблюдений]
    """
    inp    = sample['input']
    target = sample['target'][0]
    below  = sample['below']
    mask   = np.clip(inp[1] + inp[2], 0, 1)

    err_lama = pred_lama - target
    err_oi   = pred_oi   - target

    vmin, vmax = target[~below].min(), target[~below].max()
    emax = max(np.abs(err_lama[~below]).max(), np.abs(err_oi[~below]).max()) + 1e-6
    enorm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.35)

    titles = ['True T', 'LaMa pred', 'OI pred',
              'LaMa error', 'OI error', 'Observations']
    arrays = [target, pred_lama, pred_oi, err_lama, err_oi, mask]
    cmaps  = ['plasma','plasma','plasma','RdBu_r','RdBu_r','Greens']
    norms  = [None, None, None, enorm, enorm, None]
    vmins  = [vmin, vmin, vmin, None, None, 0]
    vmaxs  = [vmax, vmax, vmax, None, None, 1]

    for i, (title, arr, cmap, norm, vm, vM) in enumerate(
            zip(titles, arrays, cmaps, norms, vmins, vmaxs)):
        ax = fig.add_subplot(gs[i])
        arr_plot = arr.copy(); arr_plot[below] = np.nan
        im = ax.imshow(arr_plot, origin='upper', aspect='auto',
                       cmap=cmap, norm=norm,
                       vmin=vm, vmax=vM, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('z')

    # CTD-линии на первых трёх панелях
    ctd_xs = np.where(inp[1].sum(axis=0) > 0)[0]
    for ax_idx in [0, 1, 2]:
        ax = fig.axes[ax_idx]
        for xi in ctd_xs:
            ax.axvline(xi, color='cyan', lw=0.7, alpha=0.6)

    m_str = (f"LaMa RMSE={metrics_lama['rmse']:.4f} MAE={metrics_lama['mae']:.4f} "
             f"PSNR={metrics_lama['psnr']:.1f}dB\n"
             f"OI   RMSE={metrics_oi['rmse']:.4f}  MAE={metrics_oi['mae']:.4f}  "
             f"PSNR={metrics_oi['psnr']:.1f}dB")
    fig.suptitle(f"Epoch {epoch}\n{m_str}", fontsize=9)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for split, color in [('train','steelblue'), ('val','tomato')]:
        totals = [h['total'] for h in history[split]]
        axes[0].plot(totals, label=split, color=color)
    axes[0].set(title='Total loss', xlabel='epoch', ylabel='loss')
    axes[0].legend(); axes[0].set_yscale('log')

    components = ['data', 'stab', 'bbl', 'grad']
    for comp in components:
        vals = [h[comp] for h in history['val']]
        axes[1].plot(vals, label=comp)
    axes[1].set(title='Val loss components', xlabel='epoch', ylabel='loss')
    axes[1].legend(); axes[1].set_yscale('log')

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8. ЦИКЛ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optim, cfg, device):
    model.train()
    hist = dict(total=0, data=0, stab=0, bbl=0, grad=0)
    for inp, tgt, bathy, below in loader:
        inp, tgt, bathy, below = (t.to(device) for t in (inp, tgt, bathy, below))
        pred = model(inp)
        loss, parts = compute_loss(pred, tgt, inp, bathy, below, cfg)
        optim.zero_grad(); loss.backward(); optim.step()
        for k in hist: hist[k] += parts[k]
    n = len(loader)
    return {k: v/n for k, v in hist.items()}


@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    hist = dict(total=0, data=0, stab=0, bbl=0, grad=0)
    for inp, tgt, bathy, below in loader:
        inp, tgt, bathy, below = (t.to(device) for t in (inp, tgt, bathy, below))
        pred = model(inp)
        _, parts = compute_loss(pred, tgt, inp, bathy, below, cfg)
        for k in hist: hist[k] += parts[k]
    n = len(loader)
    return {k: v/n for k, v in hist.items()}


def run_evaluation(model, val_ds, cfg, device, epoch, prefix=""):
    """Считает метрики LaMa vs OI на первых 8 сэмплах val_ds."""
    model.eval()
    rows_lama, rows_oi = [], []

    with torch.no_grad():
        for i in range(min(8, len(val_ds))):
            s    = val_ds.samples[i]
            inp_t = torch.from_numpy(s['input']).unsqueeze(0).to(device)
            pred  = model(inp_t)[0, 0].cpu().numpy()
            tgt   = s['target'][0]
            below = s['below']

            oi    = optimal_interpolation(s['input'], cfg['NX'], cfg['NZ'])

            ml = compute_metrics(pred, tgt, below)
            mo = compute_metrics(oi,   tgt, below)
            rows_lama.append(ml); rows_oi.append(mo)

            if i == 0:
                plot_comparison(s, pred, oi, ml, mo, epoch,
                                OUT / f"{prefix}comparison_ep{epoch:04d}.png")

    def mean_metrics(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    return mean_metrics(rows_lama), mean_metrics(rows_oi)


def main():
    cfg    = CFG
    device = cfg['DEVICE']

    # данные
    print("Generating synthetic datasets...")
    train_ds = OceanDataset(cfg['N_SAMPLES_TRAIN'], cfg, seed=42)
    val_ds   = OceanDataset(cfg['N_SAMPLES_VAL'],   cfg, seed=99)
    train_dl = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'],
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg['BATCH_SIZE'],
                          shuffle=False, num_workers=0, pin_memory=False)

    # модель
    model = LamaGenerator(in_ch=4,
                          base_ch=cfg['BASE_CH'],
                          n_blocks=cfg['N_BLOCKS']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LaMa-lite params: {n_params/1e3:.1f}K")

    optim    = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, patience=cfg['LR_PATIENCE'], factor=0.5)

    history  = dict(train=[], val=[])
    best_val = math.inf
    es_count = 0

    # сводная таблица метрик
    eval_table = []

    t0 = time.time()
    for ep in range(1, cfg['EPOCHS'] + 1):
        tr = train_epoch(model, train_dl, optim, cfg, device)
        vl = val_epoch(model, val_dl, cfg, device)
        history['train'].append(tr); history['val'].append(vl)
        sched.step(vl['total'])

        improved = vl['total'] < best_val
        if improved:
            best_val = vl['total']
            es_count = 0
            torch.save({'epoch': ep, 'state': model.state_dict(),
                        'val_loss': best_val},
                       CKPT / 'best.pt')
        else:
            es_count += 1

        if ep % 10 == 0 or ep == 1:
            elapsed = (time.time() - t0) / 60
            print(f"Ep {ep:4d}/{cfg['EPOCHS']} | "
                  f"train={tr['total']:.4f} val={vl['total']:.4f} | "
                  f"stab={vl['stab']:.4f} bbl={vl['bbl']:.4f} | "
                  f"lr={optim.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}min | ES={es_count}/{cfg['ES_PATIENCE']}")

        if ep % cfg['VIZ_EVERY'] == 0 or ep == 1:
            ml, mo = run_evaluation(model, val_ds, cfg, device, ep)
            eval_table.append(dict(epoch=ep,
                                   lama_rmse=ml['rmse'], lama_mae=ml['mae'],
                                   lama_psnr=ml['psnr'],
                                   oi_rmse=mo['rmse'],   oi_mae=mo['mae'],
                                   oi_psnr=mo['psnr']))
            print(f"  → LaMa RMSE={ml['rmse']:.4f} PSNR={ml['psnr']:.1f}dB | "
                  f"OI RMSE={mo['rmse']:.4f} PSNR={mo['psnr']:.1f}dB")

        if es_count >= cfg['ES_PATIENCE']:
            print(f"Early stopping at epoch {ep}"); break

    # финальная оценка
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(CKPT / 'best.pt',
                                     map_location=device)['state'])
    ml_final, mo_final = run_evaluation(model, val_ds, cfg, device,
                                        ep, prefix="final_")
    eval_table.append(dict(epoch=-1,
                           lama_rmse=ml_final['rmse'], lama_mae=ml_final['mae'],
                           lama_psnr=ml_final['psnr'],
                           oi_rmse=mo_final['rmse'],   oi_mae=mo_final['mae'],
                           oi_psnr=mo_final['psnr']))

    # сохранение результатов
    plot_loss_curves(history, OUT / 'loss_curves.png')
    with open(OUT / 'eval_table.json', 'w') as f:
        json.dump(eval_table, f, indent=2)

    # финальная таблица в терминал
    print("\n" + "="*70)
    print(f"{'Epoch':>6} | {'LaMa RMSE':>10} {'LaMa MAE':>9} {'LaMa PSNR':>10} "
          f"| {'OI RMSE':>9} {'OI MAE':>8} {'OI PSNR':>9}")
    print("-"*70)
    for r in eval_table:
        ep_str = "BEST" if r['epoch'] == -1 else str(r['epoch'])
        print(f"{ep_str:>6} | {r['lama_rmse']:>10.4f} {r['lama_mae']:>9.4f} "
              f"{r['lama_psnr']:>10.1f} | {r['oi_rmse']:>9.4f} "
              f"{r['oi_mae']:>8.4f} {r['oi_psnr']:>9.1f}")
    print("="*70)

    total_min = (time.time() - t0) / 60
    print(f"\nDone in {total_min:.1f} min. Outputs: {OUT}")


if __name__ == "__main__":
    main()
