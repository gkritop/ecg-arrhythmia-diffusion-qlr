import copy
import glob
import math
import os
import platform
import random
import shutil
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pennylane as qml
import seaborn as sns
import wfdb
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, detrend, filtfilt
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

BASE_DIR = os.path.join(os.getcwd(), "data")
MITDB_PATH = os.path.join(BASE_DIR, "mit_bih")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MITDB_PATH, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Data Directory: {MITDB_PATH}")
print(f"Output Directory: {OUT_DIR}")

# Multi-seed configuration
ENABLE_MULTI_SEED = True
GLOBAL_SEEDS = [42, 88, 123, 456, 789, 13, 7, 99, 314, 1024] if ENABLE_MULTI_SEED else [42]

# Hyperparameters
BATCH_SIZE = 256
NUM_WORKERS = 0
PIN_MEMORY = False

LATENT_DIM = 32
VAE_EPOCHS = 200
VAE_LR = 1e-3
VAE_BETA = 0.05
VAE_PATIENCE = 20

DDPM_EPOCHS = 200
DDPM_PATIENCE = 30
DDPM_LR = 1e-4
DDPM_TIMESTEPS = 1000

CFG_DROPOUT = 0.1

N_QUBITS = 8
N_LAYERS = 6

QLR_EPOCHS = 80
QLR_LR = 2e-3
QLR_ALPHA = 0.1

POOL_SIZE = 2048
POOL_REFRESH_EVERY = 5
PATIENCE = 20
MIN_SPACING_FRAC = 0.5

CLF_EPOCHS = 100
CLF_LR = 1e-3
CLF_PATIENCE = 20

METHOD_STYLES = {
    "Baseline": {"color": "#333333", "marker": "x",  "linestyle": "-"},
    "SMOTE":    {"color": "#E69F00", "marker": "o",  "linestyle": "-"},
    "cVAE":     {"color": "#0072B2", "marker": "s",  "linestyle": "--"},
    "DDPM":     {"color": "#D55E00", "marker": "^",  "linestyle": "-."},
    "DDPM+QLR": {"color": "#009E73", "marker": "D",  "linestyle": ":"},
}

# Ablation: two classical refiner sizes for comparison against QLR (689 params:
# 144 quantum circuit + 545 gate head). Neither is an exact parameter match —
# see comments below and QLR gate-related notes.
ABLATION_MLP_CONFIGS = {
    "MLP-S": [4],    # ~300 params
    "MLP-M": [16],   # ~1100 params
}

# Global timing log
timing_log = {}

FS = 360
RPEAK_WINDOW_PRE = 0.25
RPEAK_WINDOW_POST = 0.83

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("Device:", device)

CLASS_ORDER = ["N", "S", "V", "F"]
lab2idx = {c: i for i, c in enumerate(CLASS_ORDER)}
idx2lab = {i: c for c, i in lab2idx.items()}
cond_dim = len(CLASS_ORDER)

CLASS_COLORS = {
    "N": "#0072B2",
    "S": "#009E73",
    "V": "#D55E00",
    "F": "#CC79A7",
}

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_data_exists(path):
    if len(glob.glob(os.path.join(path, "*.hea"))) == 0:
        print(f"Downloading MIT-BIH Arrhythmia Database to: {path}")
        try:
            wfdb.dl_database("mitdb", path)
            print("Download complete.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not download data. {e}")
            exit(1)
    else:
        print("MIT-BIH data found. Skipping download.")

def set_publication_style(use_latex=True, font_scale=1.0):
    plt.rcdefaults()
    sns.set_context("paper", font_scale=font_scale)
    
    if use_latex:
        font_params = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.sans-serif": ["Computer Modern Sans serif"],
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        }
    else:
        font_params = {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    
    params = {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelpad": 4,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "grid.color": "#F0F0F0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    }
    
    plt.rcParams.update(font_params)
    plt.rcParams.update(params)

SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.2

has_latex = shutil.which("latex") is not None
set_publication_style(use_latex=has_latex)
if not has_latex:
    print("WARNING: LaTeX not found. Reverting to standard fonts.")

ensure_data_exists(MITDB_PATH)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def butter_bandpass_filter(sig, fs, low=0.5, high=45.0, order=2):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def list_records(mitdb_path):
    recs = [
        os.path.splitext(os.path.basename(p))[0]
        for p in sorted(glob.glob(os.path.join(mitdb_path, "*.dat")))
    ]
    return [r for r in recs if r not in ("102", "104", "107", "217")]

def load_record(record, fs, path):
    sig = wfdb.rdrecord(os.path.join(path, record))
    ann = wfdb.rdann(os.path.join(path, record), "atr")
    prefer = ["MLII", "II", "V5", "V1"]
    names = [c.strip() for c in sig.sig_name]
    ch_idx = next(
        (i for p in prefer if p in names for i, n in enumerate(names) if n == p), 0
    )
    x = sig.p_signal[:, ch_idx].astype(np.float32)
    return butter_bandpass_filter(x, fs), ann

def segment_beats(x, ann, fs, pre_s, post_s):
    PRE_SAMPLES = int(round(pre_s * fs))
    POST_SAMPLES = int(round(post_s * fs))
    T_FIXED_LEN = PRE_SAMPLES + POST_SAMPLES
    
    segs, labs = [], []
    idxs = np.asarray(ann.sample, dtype=int)
    syms = np.asarray(ann.symbol, dtype=object)
    
    mask = np.isin(syms, list(AAMI_MAP.keys()))
    idxs, syms = idxs[mask], syms[mask]
    
    for idx, sym in zip(idxs, syms):
        lab = AAMI_MAP[sym]
        if lab == "Q":
            continue
        
        r_l = max(0, idx - int(round(0.04 * fs)))
        r_r = min(len(x), idx + int(round(0.04 * fs)) + 1)
        if r_r - r_l < 3:
            continue
        
        local = x[r_l:r_r]
        if len(local) == 0:
            continue
        
        r_off = np.argmax(local) if x[idx] > 0 else np.argmin(local)
        r_idx = r_l + r_off
        
        s_ideal = r_idx - PRE_SAMPLES
        e_ideal = r_idx + POST_SAMPLES
        
        s_data = max(0, s_ideal)
        e_data = min(len(x), e_ideal)
        
        seg_var = x[s_data:e_data].astype(np.float32)
        
        if len(seg_var) > 10:
            seg_var = detrend(seg_var, type="linear")
        
        pad_start = int(max(0, 0 - s_ideal))
        pad_end = int(max(0, e_ideal - len(x)))
        
        seg = np.pad(
            seg_var, (pad_start, pad_end), mode="constant", constant_values=0.0
        ).astype(np.float32)
        
        seg = np.clip(seg, -5.0, 5.0)
        
        segs.append(seg)
        labs.append(lab)
    
    if len(segs) == 0:
        return np.empty((0, T_FIXED_LEN), np.float32), np.empty((0,), object)
    return np.stack(segs).astype(np.float32), np.array(labs, dtype=object)

def load_stratified_temporal_split(record_list, fs, path):
    """
    - First 80% -> Train (The Past)
    - Next 10%  -> Validation (The Near Future)
    - Last 10%  -> Test (The Far Future)
    
    This creates a Fixed, Deterministic split that is valid 
    for a personal wearable device.
    """
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []
    
    print(f"Loading {len(record_list)} records with Fixed 80/10/10 Temporal Split...")
    
    for rec in record_list:
        sig, ann = load_record(rec, fs, path)
        segs, labs = segment_beats(sig, ann, fs, pre_s=0.25, post_s=0.83)
        
        n_beats = len(segs)
        if n_beats > 10:
            idx_train = int(n_beats * 0.80)
            idx_val   = int(n_beats * 0.90)
            
            X_train_list.append(segs[:idx_train])
            y_train_list.append(labs[:idx_train])
            
            X_val_list.append(segs[idx_train:idx_val])
            y_val_list.append(labs[idx_train:idx_val])

            X_test_list.append(segs[idx_val:])
            y_test_list.append(labs[idx_val:])
            
    data = {
        "X_train": np.vstack(X_train_list),
        "y_train": np.concatenate(y_train_list),
        "X_val":   np.vstack(X_val_list),
        "y_val":   np.concatenate(y_val_list),
        "X_test":  np.vstack(X_test_list),
        "y_test":  np.concatenate(y_test_list)
    }
    
    print(f"Split Complete. Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")
    return data

# =============================================================================
# CONDITIONAL VAE
# =============================================================================

class HybridEncoder(nn.Module):
    def __init__(self, input_len=389, cond_dim=4, latent_dim=32, emb_dim=8):
        super().__init__()
        self.class_emb = nn.Embedding(cond_dim, emb_dim)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.conv_out_flat_dim = 24
        self.dense_branch = nn.Sequential(
            nn.Linear(input_len, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )
        self.merge_dim = self.conv_out_flat_dim + 50 + emb_dim
        self.fc_mu = nn.Linear(self.merge_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.merge_dim, latent_dim)
    
    def forward(self, x, labels):
        c_emb = self.class_emb(labels)
        x_conv = self.conv_branch(x).view(x.size(0), -1)
        x_dense = self.dense_branch(x.view(x.size(0), -1))
        merged = torch.cat([x_conv, x_dense, c_emb], dim=1)
        return self.fc_mu(merged), self.fc_logvar(merged)

class HybridDecoder(nn.Module):
    def __init__(self, input_len=389, cond_dim=4, latent_dim=32, emb_dim=8):
        super().__init__()
        self.input_len = input_len
        self.class_emb = nn.Embedding(cond_dim, emb_dim)
        self.conv_start_len = 24
        input_dim = latent_dim + emb_dim
        
        self.conv_project = nn.Linear(input_dim, self.conv_start_len)
        self.upsample_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )
        self.dense_project = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, input_len),
        )
        self.final_dense = nn.Linear(input_len * 2, input_len)
    
    def forward(self, z, labels):
        c_emb = self.class_emb(labels)
        z_cond = torch.cat([z, c_emb], dim=1)
        
        x_conv = self.conv_project(z_cond).view(z_cond.size(0), 1, -1)
        x_conv = self.upsample_branch(x_conv)
        
        if x_conv.size(2) < self.input_len:
            x_conv = F.pad(x_conv, (0, self.input_len - x_conv.size(2)))
        elif x_conv.size(2) > self.input_len:
            x_conv = x_conv[..., :self.input_len]
        
        x_conv_flat = x_conv.view(x_conv.size(0), -1)
        x_dense = self.dense_project(z_cond)
        
        combined = torch.cat([x_conv_flat, x_dense], dim=1)
        out = self.final_dense(combined)
        return out.unsqueeze(1)

class CVAE(nn.Module):
    def __init__(self, input_len=389, cond_dim=4, latent_dim=32):
        super().__init__()
        self.enc = HybridEncoder(input_len, cond_dim, latent_dim)
        self.dec = HybridDecoder(input_len, cond_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, logvar = self.enc(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.dec(z, labels)
        return recon_x, mu, logvar


def loss_fn_spectral(recon_x, x, mu, logvar, beta=VAE_BETA):
    mse_loss = F.mse_loss(recon_x, x, reduction="mean")

    x_fft = torch.fft.rfft(x, dim=2)
    recon_fft = torch.fft.rfft(recon_x, dim=2)
    spectral_loss = F.l1_loss(torch.abs(recon_fft), torch.abs(x_fft), reduction="mean")

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    total_loss = mse_loss + spectral_loss + (beta * kld_loss)
    return total_loss, mse_loss, kld_loss

def train_vae(model, train_loader, val_loader, epochs=VAE_EPOCHS, lr=VAE_LR, patience=VAE_PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses = []
    val_losses = []

    target_beta = VAE_BETA
    anneal_epochs = 40

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"Starting VAE training for {epochs} epochs (target_beta={target_beta}, patience={patience})...")

    for epoch in range(epochs):
        current_beta = (
            target_beta * (epoch / anneal_epochs)
            if epoch < anneal_epochs
            else target_beta
        )

        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(X_batch, y_batch)

            loss, _, _ = loss_fn_spectral(
                recon, X_batch, mu, logvar, beta=current_beta
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)

                recon_v, mu_v, logvar_v = model(X_v, y_v)
                loss_v, _, _ = loss_fn_spectral(
                    recon_v, X_v, mu_v, logvar_v, beta=current_beta
                )
                running_val_loss += loss_v.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Beta: {current_beta:.4f} | "
                f"Patience: {patience_counter}/{patience}"
            )

        if epoch == anneal_epochs - 1:
            best_val_loss = float("inf")
            patience_counter = 0

        if patience_counter >= patience and epoch >= anneal_epochs:
            print(f"[VAE] Early stopping at epoch {epoch + 1} (best val={best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses

@torch.no_grad()
def encode_mu(vae, X, y_idx):
    vae.eval()
    Z = []
    bs = 512
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32).unsqueeze(1).to(device)
        yb = torch.tensor(y_idx[i:i + bs], dtype=torch.long).to(device)
        mu, _ = vae.enc(xb, yb)
        Z.append(mu.cpu().numpy())
    return np.vstack(Z)


@torch.no_grad()


# =============================================================================
# LATENT DDPM
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    
    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half, 1)
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)

class AdaLNZeroBlock(nn.Module):
    def __init__(self, width, cond_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, 3 * width) 
        )
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)

    def forward(self, x, cond_emb):
        shift, scale, gate = self.ada_mlp(cond_emb).chunk(3, dim=1)
        x_norm = self.norm(x) * (1 + scale) + shift
        return x + gate * self.net(x_norm)

class LatentMLPDenoiser(nn.Module):
    def __init__(self, cond_dim=4, latent_dim=32, width=256, emb_dim=16):
        super().__init__()
        self.class_emb = nn.Embedding(cond_dim + 1, emb_dim)

        self.time_mlp = nn.Sequential(
            TimeEmbedding(width),
            nn.Linear(width, width),
            nn.SiLU()
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, width),
            nn.SiLU()
        )

        self.input_proj = nn.Linear(latent_dim, width)
        self.z_upsampler = nn.Linear(latent_dim, width)

        self.res_layers = nn.ModuleList([
            AdaLNZeroBlock(width, cond_emb_dim=width) for _ in range(6)
        ])

        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_ada = nn.Linear(width, 2 * width)

        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)

        self.final_proj = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, latent_dim)
        )

    def forward(self, z, t, labels):
        t_emb = self.time_mlp(t.float())
        c_emb = self.cond_mlp(self.class_emb(labels))

        cond_signal = t_emb + c_emb

        x = self.input_proj(z) + self.z_upsampler(z)

        for block in self.res_layers:
            x = block(x, cond_signal)

        shift, scale = self.final_ada(cond_signal).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale) + shift

        return self.final_proj(x)

class LatentDDPM:
    def __init__(self, model, latent_dim, timesteps=1000):
        self.model = model.to(device)
        self.T = timesteps
        self.latent_dim = latent_dim
        
        scale = 1.0
        self.betas = torch.linspace(1e-4, 0.01 * scale, timesteps, device=device)
        
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
    
    def loss(self, z0, labels):
        z0 = z0.squeeze(1)
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(z0)
        
        if self.model.training and CFG_DROPOUT > 0:
            mask = torch.rand(B, device=device) < CFG_DROPOUT
            null_token = cond_dim 
            labels = torch.where(mask, torch.full_like(labels, null_token), labels)
        
        a_bar = self.alpha_bars[t][:, None]
        
        zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise
        pred_noise = self.model(zt, t, labels)
        
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def sample(self, n, labels, cfg_scale=2.0):
        self.model.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        
        null_labels = torch.full((n,), cond_dim, device=device, dtype=torch.long)
        
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            beta = self.betas[ti]
            a = self.alphas[ti]
            a_bar = self.alpha_bars[ti]
            
            # CFG Sampling
            noise_cond = self.model(z, t, labels)
            noise_uncond = self.model(z, t, null_labels)
            
            pred_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            
            # Standard Ancestral Sampling
            mean = (1 / torch.sqrt(a)) * (
                z - ((1 - a) / torch.sqrt(1 - a_bar)) * pred_noise
            )
            
            if ti > 0:
                noise = torch.randn_like(z)
                # Posterior variance
                beta_tilde = beta * (1.0 - self.alpha_bars_prev[ti]) / (1.0 - a_bar)
                z = mean + torch.sqrt(beta_tilde) * noise
            else:
                z = mean
            
            if ti % 250 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.model.train()
        return z

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.shadow = copy.deepcopy(model)
        self.decay = decay
    
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    shadow_param = self.shadow.state_dict()[name]
                    shadow_param.mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
    
    def apply_shadow(self):
        self.backup = copy.deepcopy(self.model)
        self.model.load_state_dict(self.shadow.state_dict())
    
    def restore(self):
        self.model.load_state_dict(self.backup.state_dict())

def train_latent_ddpm(ddpm, ds, epochs=200, bs=256, lr=1e-4, sampler=None, val_ds=None, patience=DDPM_PATIENCE):
    model = ddpm.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=0.9999)
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS) if val_ds is not None else None

    train_history = []
    val_history = []
    best_val_loss = float("inf")
    ddpm_patience_counter = 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            
            loss = ddpm.loss(z, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ema.update()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        train_history.append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for z_v, y_v in val_loader:
                    z_v, y_v = z_v.to(device), y_v.to(device)
                    # Temporarily disable CFG dropout for clean val loss
                    z_v = z_v.squeeze(1)
                    B = z_v.size(0)
                    t = torch.randint(0, ddpm.T, (B,), device=device)
                    noise = torch.randn_like(z_v)
                    a_bar = ddpm.alpha_bars[t][:, None]
                    zt = torch.sqrt(a_bar) * z_v + torch.sqrt(1 - a_bar) * noise
                    pred_noise = model(zt, t, y_v)
                    val_losses.append(F.mse_loss(pred_noise, noise).item())
            current_val = np.mean(val_losses)
            val_history.append(current_val)
            model.train()

            if current_val < best_val_loss:
                best_val_loss = current_val
                ddpm_patience_counter = 0
            else:
                ddpm_patience_counter += 1

        if ep % 10 == 0:
            val_str = f" | Val Loss={val_history[-1]:.4f}" if val_history else ""
            pat_str = f" | Patience={ddpm_patience_counter}/{patience}" if val_history else ""
            print(f"[Latent DDPM] ep {ep:03d}/{epochs} avg loss={avg_loss:.4f}{val_str}{pat_str}")

        if val_history and ddpm_patience_counter >= patience:
            print(f"[DDPM] Early stopping at ep {ep} (best val={best_val_loss:.4f})")
            break

    ema.apply_shadow()
    return ddpm, train_history, val_history

# =============================================================================
# QUANTUM LATENT REFINEMENT (QLR)
# =============================================================================

dev_q = qml.device("default.qubit", wires=N_QUBITS)

def get_robust_bounds(data_numpy, low_p=1, high_p=99):
    low  = np.percentile(data_numpy, low_p,  axis=0)
    high = np.percentile(data_numpy, high_p, axis=0)
    return (
        torch.tensor(low,  dtype=torch.float32),
        torch.tensor(high, dtype=torch.float32),
    )

def to_unit_robust(z, low, high):
    low  = low.to(z.device)
    high = high.to(z.device)
    z_clipped = torch.clamp(z, low, high)
    return 2.0 * (z_clipped - low) / (high - low + 1e-8) - 1.0

def from_unit_robust(z_unit, low, high):
    low  = low.to(z_unit.device)
    high = high.to(z_unit.device)
    return 0.5 * (z_unit + 1.0) * (high - low + 1e-8) + low

def embed_features(x_cpu):
    for q in range(N_QUBITS):
        i = q * 4

        qml.RY(x_cpu[:, i + 0] * math.pi, wires=q)
        qml.RZ(x_cpu[:, i + 1] * math.pi, wires=q)
        qml.RY(x_cpu[:, i + 2] * math.pi, wires=q)
        qml.RZ(x_cpu[:, i + 3] * math.pi, wires=q)

def entangle_layer(layer_idx):
    if layer_idx % 2 == 0:
        pairs = [(q, q + 1) for q in range(0, N_QUBITS - 1, 2)]
    else:
        pairs = [(q, q + 1) for q in range(1, N_QUBITS - 1, 2)]
        pairs.append((N_QUBITS - 1, 0))

    for ctrl, tgt in pairs:
        qml.CNOT(wires=[ctrl, tgt])

    if layer_idx % 3 == 2:
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 2) % N_QUBITS])

@qml.qnode(dev_q, interface="torch", diff_method="backprop")
def qnode(params, features_cpu):
    embed_features(features_cpu)

    for l in range(params.shape[0]):
        for q in range(N_QUBITS):
            qml.Rot(*params[l, q], wires=q)

        entangle_layer(l)

    z_exp  = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    x_exp  = [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
    y_exp  = [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)]
    zz_exp = [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % N_QUBITS)) for i in range(N_QUBITS)]

    return z_exp + x_exp + y_exp + zz_exp

def loss_stay(z_ref, z_orig):
    return F.mse_loss(z_ref, z_orig, reduction="mean")


def mmd_loss(x, y, max_samples=1000, bandwidth_mults=(0.5, 1.0, 2.0)):
    if x.size(0) > max_samples:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_samples]]
    if y.size(0) > max_samples:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_samples]]

    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2

    all_dists = torch.cat([xx.reshape(-1), yy.reshape(-1), xy.reshape(-1)])
    all_dists = all_dists[all_dists > 1e-6]
    base_bandwidth = (
        1.0 / (torch.median(all_dists) + 1e-8)
        if len(all_dists) > 0
        else torch.tensor(1.0, device=x.device)
    )

    mmd_total = 0.0
    for mult in bandwidth_mults:
        bandwidth = base_bandwidth * mult
        K_xx = torch.exp(-bandwidth * xx).mean()
        K_yy = torch.exp(-bandwidth * yy).mean()
        K_xy = torch.exp(-bandwidth * xy).mean()
        mmd_total = mmd_total + (K_xx + K_yy - 2.0 * K_xy)

    return mmd_total / len(bandwidth_mults)


def qlr_distribution_loss(z_refined, z_orig, z_real, w_stay=1.0, w_mmd=1.0):
    
    loss_stay_term = loss_stay(z_refined, z_orig)
    loss_mmd_term = mmd_loss(z_refined, z_real)
    loss_total = w_stay * loss_stay_term + w_mmd * loss_mmd_term

    return loss_total, loss_stay_term, loss_mmd_term

class LatentClassifier(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class QuantumRefiner(nn.Module):
    def __init__(
        self,
        n_layers = N_LAYERS,
        alpha = QLR_ALPHA,
        lr = QLR_LR,
        minority_only = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.alpha = alpha
        self.lr = lr
        self.minority_only = minority_only

        self.params = nn.ParameterDict({
            lab: nn.Parameter(
                torch.randn(n_layers, N_QUBITS, 3, device="cpu") * 0.1
            )
            for lab in ["S", "V", "F"]
        })

        self.gate = nn.ModuleDict()
        for lab in ["S", "V", "F"]:
            head = nn.Sequential(
                nn.Linear(LATENT_DIM, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(head[-2].weight)
            nn.init.constant_(head[-2].bias, 4.0)
            self.gate[lab] = head

        for lab in ["S", "V", "F"]:
            self.register_buffer(f"z_min_{lab}", torch.zeros(LATENT_DIM))
            self.register_buffer(f"z_max_{lab}", torch.ones(LATENT_DIM))

    def _assert_cpu_param(self, lab):
        assert self.params[lab].device.type == "cpu", (
            f"QLR parameters for class {lab} must remain on CPU, "
            f"got {self.params[lab].device}."
        )

    def circuit_delta(self, lab, x_unit_cpu):
        self._assert_cpu_param(lab)
        assert x_unit_cpu.device.type == "cpu", "QLR circuit inputs must be on CPU."

        raw = torch.stack(
            qnode(self.params[lab], x_unit_cpu), dim=1
        ).float()

        mean = raw.mean(dim=1, keepdim=True)
        std  = raw.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        direction = (raw - mean) / std

        gate = self.gate[lab](direction)
        return direction * gate

    def get_bounds(self, lab):
        low  = getattr(self, f"z_min_{lab}")
        high = getattr(self, f"z_max_{lab}")
        return low, high

    def sample_pool(self, lab, ddpm, mu_z, std_z, low, high, pool_size):
        ddpm_device = next(ddpm.model.parameters()).device
        low = low.to(ddpm_device)
        high = high.to(ddpm_device)
        c_labels = torch.full((pool_size,), lab2idx[lab], device=ddpm_device, dtype=torch.long)

        with torch.no_grad():
            z_norm = ddpm.sample(pool_size, c_labels)
            z_vae = z_norm * std_z.to(ddpm_device) + mu_z.to(ddpm_device)
            pool_unit = to_unit_robust(z_vae, low, high)

        return pool_unit.cpu()

    def fit_class(self, lab, real_bank_vae, ddpm, mu_z, std_z, iters, bs):
        self._assert_cpu_param(lab)

        low, high = get_robust_bounds(real_bank_vae.cpu().numpy())
        getattr(self, f"z_min_{lab}").copy_(low)
        getattr(self, f"z_max_{lab}").copy_(high)

        real_bank_cpu = real_bank_vae.cpu()

        print(f"[QLR {lab}] alpha={self.alpha}")

        opt = torch.optim.Adam(
            [self.params[lab], *self.gate[lab].parameters()], lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=iters, eta_min=5e-4
        )

        pool_unit_cpu = self.sample_pool(
            lab, ddpm, mu_z, std_z, low, high, POOL_SIZE
        )

        best_val_loss = float("inf")
        best_state = self.params[lab].detach().clone()
        best_gate_state = copy.deepcopy(self.gate[lab].state_dict())
        patience_counter = 0
        patience = PATIENCE

        pbar = tqdm(range(iters), desc=f"QLR {lab}", leave=False)
        for epoch in pbar:
            if epoch > 0 and epoch % POOL_REFRESH_EVERY == 0:
                pool_unit_cpu = self.sample_pool(
                    lab, ddpm, mu_z, std_z, low, high, POOL_SIZE
                )

            perm = torch.randperm(POOL_SIZE)
            n_batches = POOL_SIZE // bs
            epoch_total = 0.0
            epoch_stay = 0.0
            epoch_mmd = 0.0

            for i in range(n_batches):
                idx = perm[i * bs : (i + 1) * bs]
                batch_cpu = pool_unit_cpu[idx]

                opt.zero_grad()

                delta = self.circuit_delta(lab, batch_cpu)
                refined_unit = torch.clamp(batch_cpu + self.alpha * delta, -1.0, 1.0)
                z_orig = from_unit_robust(batch_cpu, low, high)
                z_refined = from_unit_robust(refined_unit, low, high)

                loss_total, loss_stay_term, loss_mmd_term = qlr_distribution_loss(
                    z_refined, z_orig, real_bank_cpu
                )

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [self.params[lab], *self.gate[lab].parameters()], max_norm=1.0
                )
                opt.step()

                epoch_total += loss_total.item()
                epoch_stay += loss_stay_term.item()
                epoch_mmd += loss_mmd_term.item()

            sched.step()

            with torch.no_grad():
                val_sample_size = min(512, pool_unit_cpu.size(0))
                val_batch_cpu = pool_unit_cpu[:val_sample_size]
                delta_val = self.circuit_delta(lab, val_batch_cpu)
                refined_unit_val = torch.clamp(
                    val_batch_cpu + self.alpha * delta_val, -1.0, 1.0
                )
                z_orig_val = from_unit_robust(val_batch_cpu, low, high)
                z_refined_val = from_unit_robust(refined_unit_val, low, high)
                val_total, val_stay, val_mmd = qlr_distribution_loss(
                    z_refined_val, z_orig_val, real_bank_cpu
                )
                movement_dist = (z_refined_val - z_orig_val).norm(dim=1).mean().item()
                val_loss = val_total.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.params[lab].detach().clone()
                best_gate_state = copy.deepcopy(self.gate[lab].state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"\n[QLR {lab}] Early stopping at epoch {epoch}: validation loss not improving"
                    )
                    break

            if epoch % 5 == 0:
                pbar.set_postfix({
                    "Loss": f"{epoch_total / n_batches:.4f}",
                    "Stay": f"{epoch_stay / n_batches:.4f}",
                    "MMD": f"{epoch_mmd / n_batches:.4f}",
                    "Val": f"{val_loss:.4f}",
                    "Move": f"{movement_dist:.4f}",
                })

        self.params[lab].data.copy_(best_state)
        self.gate[lab].load_state_dict(best_gate_state)

    def fit(self, Z_train_mu, y_labels, ddpm, mu_z, std_z, iters, bs):
        for lab in ["S", "V", "F"]:
            indices = np.where(y_labels == lab2idx[lab])[0]
            if len(indices) == 0:
                print(f"[QLR] Skipping class {lab} — no training samples.")
                continue

            real_bank = torch.tensor(Z_train_mu[indices], dtype=torch.float32)
            self.fit_class(lab, real_bank, ddpm, mu_z, std_z, iters, bs)

    @torch.no_grad()
    def refine(self, lab, z_vae_numpy):
        if lab == "N":
            return z_vae_numpy

        self._assert_cpu_param(lab)
        low, high = self.get_bounds(lab)

        z_t = torch.tensor(z_vae_numpy, dtype=torch.float32)
        z_unit = to_unit_robust(z_t, low, high)

        refined_chunks = []
        chunk = 256

        for start in range(0, z_unit.size(0), chunk):
            batch_cpu = z_unit[start : start + chunk]
            delta = self.circuit_delta(lab, batch_cpu)
            ref_unit = torch.clamp(batch_cpu + self.alpha * delta, -1.0, 1.0)
            ref_vae = from_unit_robust(ref_unit, low, high)
            refined_chunks.append(ref_vae)

        return torch.cat(refined_chunks).cpu().numpy()


# =============================================================================
# ABLATION STUDY
# =============================================================================

class ClassicalRefiner(nn.Module):
    """Flexible classical refiner for ablation.

    hidden_dims: list of hidden layer widths, e.g. [4], [16], [32].
    Trained under identical conditions to QLR so comparisons are fair.
    """

    def __init__(self, input_dim=LATENT_DIM, hidden_dims=None, alpha=QLR_ALPHA):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4]
        self.alpha = alpha
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, input_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  [ClassicalRefiner] hidden_dims={hidden_dims}  params={n_params}")

    def forward(self, x_unit):
        delta = self.net(x_unit)
        return torch.clamp(x_unit + self.alpha * delta, -1.0, 1.0)


# =============================================================================
# MOBILENETV2 CLASSIFIER WITH BALANCED CROSS-ENTROPY
# =============================================================================

def compute_balanced_class_weights(y_indices, n_classes):
    if isinstance(y_indices, torch.Tensor):
        y_np = y_indices.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_indices)

    counts = np.bincount(y_np, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_planes),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_1D(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, width_mult=1.0):
        super().__init__()
        
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        
        self.features = [ConvBNReLU(input_channels, input_channel, stride=2)]
        
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        self.features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        )
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel * 2, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        
        avg_p = F.adaptive_avg_pool1d(x, 1).flatten(1)
        max_p = F.adaptive_max_pool1d(x, 1).flatten(1)
        
        x = torch.cat([avg_p, max_p], dim=1)
        
        x = self.classifier(x)
        return x

def train_clf(model, train_loader, val_loader, epochs=CLF_EPOCHS, patience=CLF_PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=CLF_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=7
    )

    train_targets = train_loader.dataset.tensors[1]
    class_weights = compute_balanced_class_weights(train_targets, cond_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    model.to(device)
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    iterator = tqdm(range(epochs), desc="Training Epochs")

    for ep in iterator:
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())

        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        avg_loss = train_loss / len(train_loader)
        scheduler.step(val_f1)
        iterator.set_postfix({"Loss": f"{avg_loss:.4f}", "Val_F1": f"{val_f1:.4f}",
                               "LR": f"{opt.param_groups[0]['lr']:.1e}",
                               "Patience": f"{patience_counter}/{patience}"})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Classifier] Early stopping at epoch {ep}: validation Macro F1 not improving")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del opt

    return model, best_val_f1

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            preds.extend(model(x).argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    present_indices = sorted(set(trues) | set(preds))
    present_labels = [idx2lab[i] for i in present_indices]
    report = classification_report(
        trues, preds,
        labels=present_indices,
        target_names=present_labels,
        output_dict=True, zero_division=0
    )

    flat_metrics = {
        "ACC": report["accuracy"],
        "Macro_F1": f1_score(trues, preds, average="macro", zero_division=0),
        "Macro_Recall": recall_score(trues, preds, average="macro", zero_division=0),
    }

    for c in CLASS_ORDER:
        flat_metrics[f"{c}_PPV"] = report.get(c, {}).get("precision", 0.0)
        flat_metrics[f"{c}_SEN"] = report.get(c, {}).get("recall",    0.0)
        flat_metrics[f"{c}_F1"]  = report.get(c, {}).get("f1-score",  0.0)
    
    return {"metrics": flat_metrics, "report": report, "y_true": trues, "y_pred": preds}

# =============================================================================
# GENERATION & MMD FUNCTIONS
# =============================================================================

@torch.no_grad()
def decode_latents(vae, z, lab):
    labels = torch.full((len(z),), lab2idx[lab], device=device, dtype=torch.long)
    z = torch.tensor(z, dtype=torch.float32, device=device)
    x = vae.dec(z, labels)
    return x.squeeze(1).cpu().numpy()

@torch.no_grad()
def generate_data(mode, lab, n, vae, latent_ddpm, qlr, mu_z, std_z, cfg_scale=2.0):
    if mode == "cVAE":
        z = torch.randn(n, LATENT_DIM, device=device).cpu().numpy()
    
    elif "DDPM" in mode:
        c_labels = torch.full((n,), lab2idx[lab], device=device, dtype=torch.long)
        
        z_norm = latent_ddpm.sample(n, c_labels, cfg_scale=cfg_scale).cpu().numpy()
        z = z_norm * std_z + mu_z 
        
        if mode == "DDPM+QLR" and lab != "N":
            z = qlr.refine(lab, z)
    
    return decode_latents(vae, z, lab)

def mmd_gaussian(x, y, gammas=[0.5, 1.0, 2.0, 5.0, 10.0], max_samples=2000):
    if x.shape[0] > max_samples:
        idx_x = np.random.choice(x.shape[0], max_samples, replace=False)
        x = x[idx_x]
    
    if y.shape[0] > max_samples:
        idx_y = np.random.choice(y.shape[0], max_samples, replace=False)
        y = y[idx_y]

    x_t = torch.tensor(x, device='cpu', dtype=torch.float32)
    y_t = torch.tensor(y, device='cpu', dtype=torch.float32)
    
    if x_t.dim() != 2:
        x_t = x_t.view(x_t.size(0), -1)
    if y_t.dim() != 2:
        y_t = y_t.view(y_t.size(0), -1)
    
    xx_dist = torch.cdist(x_t, x_t, p=2) ** 2
    yy_dist = torch.cdist(y_t, y_t, p=2) ** 2
    xy_dist = torch.cdist(x_t, y_t, p=2) ** 2
    
    cost = 0
    for g in gammas:
        term1 = torch.exp(-g * xx_dist).mean()
        term2 = torch.exp(-g * yy_dist).mean()
        term3 = 2 * torch.exp(-g * xy_dist).mean()
        cost += term1 + term2 - term3
        
    return (cost / len(gammas)).item()

def calculate_cosine_score(X_real, X_synth):
    X_r = X_real.reshape(len(X_real), -1)
    X_s = X_synth.reshape(len(X_synth), -1)
    
    mean_r = np.mean(X_r, axis=0).reshape(1, -1)
    mean_s = np.mean(X_s, axis=0).reshape(1, -1)
    
    score = cosine_similarity(mean_r, mean_s)[0][0]
    return score

def calculate_diversity_metrics(X_real, X_synth):
    if len(X_synth) > 1000:
        X_synth_sub = X_synth[np.random.choice(len(X_synth), 1000, replace=False)]
    else:
        X_synth_sub = X_synth
    
    div_dists = pdist(X_synth_sub.reshape(len(X_synth_sub), -1), metric="euclidean")
    diversity = np.mean(div_dists)
    
    if len(X_real) > 1000:
        X_real_sub = X_real[np.random.choice(len(X_real), 1000, replace=False)]
    else:
        X_real_sub = X_real
    
    real_dists = cdist(
        X_synth_sub.reshape(len(X_synth_sub), -1),
        X_real_sub.reshape(len(X_real_sub), -1),
        metric="cosine",
    )
    
    min_dists = np.min(real_dists, axis=1)
    fidelity = np.mean(min_dists)
    
    return diversity, fidelity

def _ablation_sample_pool(lab, ddpm_model, mu_z_t, std_z_t, low_t, high_t, pool_size):
    """Sample a fresh DDPM pool and map to unit space (CPU). Mirrors QuantumRefiner.sample_pool."""
    ddpm_device = next(ddpm_model.model.parameters()).device
    c_labels = torch.full((pool_size,), lab2idx[lab], device=ddpm_device, dtype=torch.long)
    with torch.no_grad():
        z_norm  = ddpm_model.sample(pool_size, c_labels)
        z_raw   = z_norm * std_z_t.to(ddpm_device) + mu_z_t.to(ddpm_device)
        pool_u  = to_unit_robust(z_raw, low_t.to(ddpm_device), high_t.to(ddpm_device))
    return pool_u.cpu()


def _ablation_downstream_eval(lab, refiner_name, refined_latents,
                               vae, X_train, y_train, X_val, y_val, X_test, y_test,
                               mu_z, std_z, n_synth=2000, clf_epochs=50):
    """Train a classifier on real + refined synthetic data; report downstream metrics."""
    # Decode refined latents to ECG signals
    X_synth = decode_latents(vae, refined_latents[:n_synth], lab)
    y_synth = np.array([lab] * len(X_synth))

    # Augmented train set (real + synthetic of the target class only)
    X_aug = np.concatenate([X_train, X_synth], axis=0)
    y_aug = np.concatenate([y_train, y_synth], axis=0)

    y_aug_idx = torch.tensor([lab2idx[l] for l in y_aug], dtype=torch.long)
    X_aug_t   = torch.tensor(X_aug, dtype=torch.float32).unsqueeze(1)

    aug_loader = DataLoader(TensorDataset(X_aug_t, y_aug_idx),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    X_val_t   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_idx = torch.tensor([lab2idx[l] for l in y_val], dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_idx = torch.tensor([lab2idx[l] for l in y_test], dtype=torch.long)
    val_loader  = DataLoader(TensorDataset(X_val_t,  y_val_idx),  batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_idx), batch_size=256, shuffle=False)

    clf = MobileNetV2_1D(num_classes=cond_dim, input_channels=1).to(device)
    clf, _ = train_clf(clf, aug_loader, val_loader, epochs=clf_epochs)

    res = evaluate(clf, test_loader)
    lab_idx = lab2idx[lab]
    # Per-class precision, recall, F1 for the ablation target class
    cr = res["report"]
    class_name = idx2lab[lab_idx]
    cls_metrics = {
        "F1":        res["metrics"].get("Macro_F1", float("nan")),
        f"{lab}_F1": float(cr.get(class_name, {}).get("f1-score", float("nan"))),
        f"{lab}_P":  float(cr.get(class_name, {}).get("precision", float("nan"))),
        f"{lab}_R":  float(cr.get(class_name, {}).get("recall", float("nan"))),
    }
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [{refiner_name}] Macro F1={cls_metrics['F1']:.4f}  "
          f"{lab}-F1={cls_metrics[f'{lab}_F1']:.4f}  "
          f"P={cls_metrics[f'{lab}_P']:.4f}  R={cls_metrics[f'{lab}_R']:.4f}")
    return cls_metrics


def run_ablation_comparison(
    lab, Z_real_vae, ddpm_model, z_min_t, z_max_t,
    from_unit_fn, mu_z_t, std_z_t,
    vae=None, X_train=None, y_train=None,
    X_val=None, y_val=None, X_test=None, y_test=None,
    n_iters=80, bs=128):
    
    abl_device = mu_z_t.device
    real_bank_cpu = torch.tensor(Z_real_vae, dtype=torch.float32)
    real_bank_gpu = real_bank_cpu.to(abl_device)

    low_cpu = z_min_t.cpu()
    high_cpu = z_max_t.cpu()

    # ------------------------------------------------------------------ QLR model
    q_ref = QuantumRefiner(n_layers=N_LAYERS, alpha=QLR_ALPHA, minority_only=False)
    getattr(q_ref, f"z_min_{lab}").copy_(low_cpu)
    getattr(q_ref, f"z_max_{lab}").copy_(high_cpu)
    opt_q = torch.optim.Adam(
        [q_ref.params[lab], *q_ref.gate[lab].parameters()], lr=QLR_LR
    )
    sched_q = torch.optim.lr_scheduler.CosineAnnealingLR(opt_q, T_max=n_iters, eta_min=5e-4)
    best_val_q = float("inf")
    best_state_q = q_ref.params[lab].detach().clone()
    best_gate_state_q = copy.deepcopy(q_ref.gate[lab].state_dict())

    # ------------------------------------------------------------------ MLP models
    mlp_refs = {}
    mlp_opts = {}
    mlp_scheds = {}
    mlp_best_val = {}
    mlp_best_state = {}
    for name, hdims in ABLATION_MLP_CONFIGS.items():
        print(f"  Building {name}:", end=" ")
        m = ClassicalRefiner(hidden_dims=hdims, alpha=QLR_ALPHA).to(abl_device)
        mlp_refs[name] = m
        mlp_opts[name] = torch.optim.Adam(m.parameters(), lr=QLR_LR)
        mlp_scheds[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
            mlp_opts[name], T_max=n_iters, eta_min=5e-4
        )
        mlp_best_val[name] = float("inf")
        mlp_best_state[name] = copy.deepcopy(m.state_dict())

    # ------------------------------------------------------------------ initial pool
    pool_unit_cpu = _ablation_sample_pool(
        lab, ddpm_model, mu_z_t, std_z_t, z_min_t, z_max_t, POOL_SIZE
    )

    n_batches = POOL_SIZE // bs
    val_size = min(512, POOL_SIZE)

    history = {"Quantum": [], "Quantum_val": []}
    for name in ABLATION_MLP_CONFIGS:
        history[name] = []
        history[f"{name}_val"] = []

    print(
        f"[Ablation {lab}] {n_iters} epochs | bs={bs} | "
        f"{n_batches} batches/epoch | refresh every {POOL_REFRESH_EVERY}"
    )

    pbar = tqdm(range(n_iters), desc=f"Ablation {lab}", leave=False)
    for epoch in pbar:
        if epoch > 0 and epoch % POOL_REFRESH_EVERY == 0:
            pool_unit_cpu = _ablation_sample_pool(
                lab, ddpm_model, mu_z_t, std_z_t, z_min_t, z_max_t, POOL_SIZE
            )

        pool_unit_gpu = pool_unit_cpu.to(abl_device)
        perm = torch.randperm(POOL_SIZE)

        epoch_loss_q = 0.0
        epoch_loss_mlp = {name: 0.0 for name in ABLATION_MLP_CONFIGS}

        for i in range(n_batches):
            idx = perm[i * bs : (i + 1) * bs]
            batch_cpu = pool_unit_cpu[idx]
            batch_gpu = pool_unit_gpu[idx]

            # QLR step
            opt_q.zero_grad()
            delta_q = q_ref.circuit_delta(lab, batch_cpu)
            ref_q_u = torch.clamp(batch_cpu + QLR_ALPHA * delta_q, -1.0, 1.0)
            loss_q, _, _ = qlr_distribution_loss(
                from_unit_fn(ref_q_u), from_unit_fn(batch_cpu), real_bank_cpu
            )
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(
                [q_ref.params[lab], *q_ref.gate[lab].parameters()], 1.0
            )
            opt_q.step()
            epoch_loss_q += loss_q.item()

            # MLP steps
            for name, m in mlp_refs.items():
                mlp_opts[name].zero_grad()
                ref_u = m(batch_gpu)
                loss_m, _, _ = qlr_distribution_loss(
                    from_unit_fn(ref_u), from_unit_fn(batch_gpu), real_bank_gpu
                )
                loss_m.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                mlp_opts[name].step()
                epoch_loss_mlp[name] += loss_m.item()

        sched_q.step()
        for name in ABLATION_MLP_CONFIGS:
            mlp_scheds[name].step()

        avg_q = epoch_loss_q / n_batches
        history["Quantum"].append(avg_q)
        for name in ABLATION_MLP_CONFIGS:
            history[name].append(epoch_loss_mlp[name] / n_batches)

        # ---- validation ---------------------------------------------------
        val_cpu = pool_unit_cpu[:val_size]
        val_gpu = pool_unit_gpu[:val_size]

        with torch.no_grad():
            delta_qv = q_ref.circuit_delta(lab, val_cpu)
            ref_qv_u = torch.clamp(val_cpu + QLR_ALPHA * delta_qv, -1.0, 1.0)
            val_q, _, _ = qlr_distribution_loss(
                from_unit_fn(ref_qv_u), from_unit_fn(val_cpu), real_bank_cpu
            )
            val_q = val_q.item()
            history["Quantum_val"].append(val_q)
            if val_q < best_val_q:
                best_val_q = val_q
                best_state_q = q_ref.params[lab].detach().clone()
                best_gate_state_q = copy.deepcopy(q_ref.gate[lab].state_dict())

            for name, m in mlp_refs.items():
                ref_v = m(val_gpu)
                val_m, _, _ = qlr_distribution_loss(
                    from_unit_fn(ref_v), from_unit_fn(val_gpu), real_bank_gpu
                )
                val_m = val_m.item()
                history[f"{name}_val"].append(val_m)
                if val_m < mlp_best_val[name]:
                    mlp_best_val[name] = val_m
                    mlp_best_state[name] = copy.deepcopy(m.state_dict())

    # ---- restore best parameters -----------------------------------------
    q_ref.params[lab].data.copy_(best_state_q)
    q_ref.gate[lab].load_state_dict(best_gate_state_q)
    for name, m in mlp_refs.items():
        m.load_state_dict(mlp_best_state[name])

    # ---- MMD before / after refinement for each refiner ------------------
    n_eval = min(POOL_SIZE, 1000)
    eval_pool_cpu = _ablation_sample_pool(
        lab, ddpm_model, mu_z_t, std_z_t, z_min_t, z_max_t, n_eval
    )
    eval_pool_gpu = eval_pool_cpu.to(abl_device)

    # Compare in unit space to avoid fixed-gamma kernel collapse on large-scale latents.
    # eval_pool_cpu is already in unit space; real_bank_cpu needs mapping.
    real_unit_cpu = to_unit_robust(real_bank_cpu[:n_eval], low_cpu, high_cpu)
    mmd_before = mmd_loss(real_unit_cpu, eval_pool_cpu[:n_eval]).item()
    mmd_results = {"Before": mmd_before}
    with torch.no_grad():
        delta_q = q_ref.circuit_delta(lab, eval_pool_cpu)
        ref_q_u = torch.clamp(eval_pool_cpu + QLR_ALPHA * delta_q, -1.0, 1.0)
        mmd_results["QLR"] = mmd_loss(real_unit_cpu, ref_q_u).item()
        for name, m in mlp_refs.items():
            ref_m_u = m(eval_pool_gpu).cpu()
            mmd_results[name] = mmd_loss(real_unit_cpu, ref_m_u).item()

    print(f"\n[Ablation {lab}] MMD scores (lower=better, real dist as target):")
    for k, v in mmd_results.items():
        print(f"  {k:12s}: {v:.6f}")

    history["mmd_results"] = mmd_results

    # Loss curves prove distribution objectives; downstream metrics prove
    # the gains transfer to actual classification performance.
    if vae is not None and X_train is not None:
        print(f"\n[Ablation {lab}] Downstream classification (50-epoch clf per refiner):")
        mu_z = mu_z_t.cpu().numpy()
        std_z = std_z_t.cpu().numpy()
        n_synth = min(2000, POOL_SIZE)

        downstream = {}

        # DDPM only (no refinement — baseline for this comparison)
        ddpm_latents_u = _ablation_sample_pool(
            lab, ddpm_model, mu_z_t, std_z_t, z_min_t, z_max_t, n_synth
        )
        ddpm_latents_raw = from_unit_fn(ddpm_latents_u).numpy()
        downstream["DDPM"] = _ablation_downstream_eval(
            lab, "DDPM", ddpm_latents_raw,
            vae, X_train, y_train, X_val, y_val, X_test, y_test,
            mu_z, std_z, n_synth=n_synth
        )

        # QLR-refined
        with torch.no_grad():
            delta_q = q_ref.circuit_delta(lab, ddpm_latents_u)
            ref_q_u = torch.clamp(ddpm_latents_u + QLR_ALPHA * delta_q, -1.0, 1.0)
        qlr_latents_raw = from_unit_fn(ref_q_u).numpy()
        downstream["QLR"] = _ablation_downstream_eval(
            lab, "QLR", qlr_latents_raw,
            vae, X_train, y_train, X_val, y_val, X_test, y_test,
            mu_z, std_z, n_synth=n_synth
        )

        # MLP-refined (each size)
        ddpm_latents_gpu = ddpm_latents_u.to(abl_device)
        for name, m in mlp_refs.items():
            with torch.no_grad():
                ref_m_u = m(ddpm_latents_gpu).cpu()
            mlp_latents_raw = from_unit_fn(ref_m_u).numpy()
            downstream[name] = _ablation_downstream_eval(
                lab, name, mlp_latents_raw,
                vae, X_train, y_train, X_val, y_val, X_test, y_test,
                mu_z, std_z, n_synth=n_synth
            )

        history["downstream"] = downstream

        # Summary table
        print(f"\n[Ablation {lab}] Downstream summary ({lab}-class F1):")
        print(f"  {'Refiner':20s}  {'Macro F1':>9}  {lab+'-F1':>7}  {lab+'-P':>7}  {lab+'-R':>7}")
        print("  " + "-" * 55)
        for rname, dm in downstream.items():
            print(f"  {rname:20s}  "
                  f"{dm['F1']:>9.4f}  "
                  f"{dm.get(f'{lab}_F1', float('nan')):>7.4f}  "
                  f"{dm.get(f'{lab}_P', float('nan')):>7.4f}  "
                  f"{dm.get(f'{lab}_R', float('nan')):>7.4f}")

    return history

# =============================================================================
# ENSEMBLE METHOD
# =============================================================================

def run_ensemble_evaluation(fixed_data, seeds, mode="DDPM+QLR", ratio=1.0):
    print(f"\n{'='*40}")
    print(f" ENSEMBLE EVALUATION: {mode} (Ratio {ratio})")
    print(f"{'='*40}")

    X_test_t = torch.tensor(fixed_data["X_test"], dtype=torch.float32).unsqueeze(1)
    y_test_idx = torch.tensor([lab2idx[l] for l in fixed_data["y_test"]], dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_idx), batch_size=256, shuffle=False)

    models = []
    for seed in seeds:
        model_path = os.path.join(OUT_DIR, f"model_{mode}_r{ratio}_seed{seed}.pth")
        
        if os.path.exists(model_path):
            m = MobileNetV2_1D(num_classes=cond_dim, input_channels=1).to(device)
            m.load_state_dict(torch.load(model_path))
            m.eval()
            models.append(m)
            print(f"Loaded: {model_path}")
        else:
            print(f"Warning: Missing model {model_path}")

    if not models:
        print("No models found for ensembling.")
        return

    all_preds, all_trues = [], []
    
    print("Running inference on ensemble...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            batch_probs = torch.stack([F.softmax(m(x), dim=1) for m in models])
            
            avg_probs = torch.mean(batch_probs, dim=0) 
            
            all_preds.extend(avg_probs.argmax(1).cpu().numpy())
            all_trues.extend(y.numpy())

    print("\n>>> ENSEMBLE CLASSIFICATION REPORT")
    print(classification_report(all_trues, all_preds, target_names=CLASS_ORDER, digits=4))
    
    f1 = f1_score(all_trues, all_preds, average="macro")
    print(f"Ensemble Macro F1: {f1:.4f}")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_random_beats(X, y, title, out_name):
    fig_width = DOUBLE_COLUMN
    fig_height = fig_width * 0.6
    
    fig, axes = plt.subplots(
        len(CLASS_ORDER), 3, figsize=(fig_width, fig_height), sharex=True, sharey="row"
    )
    
    for r, lab in enumerate(CLASS_ORDER):
        idxs = np.where(np.array(y) == lab)[0]
        
        if len(idxs) == 0:
            for j in range(3):
                axes[r, j].axis("off")
            continue
        
        n_select = min(3, len(idxs))
        pick = np.random.choice(idxs, size=n_select, replace=False)
        selected_beats = X[pick]
        
        row_min = selected_beats.min()
        row_max = selected_beats.max()
        val_range = row_max - row_min
        
        pad = val_range * 0.1 if val_range != 0 else 0.1
        y_lim_bottom = row_min - pad
        y_lim_top = row_max + pad
        
        for j in range(3):
            ax = axes[r, j]
            
            if j < len(pick):
                p = pick[j]
                ax.plot(X[p], color=CLASS_COLORS[lab], alpha=0.9)
                ax.fill_between(
                    range(len(X[p])), X[p], alpha=0.05, color=CLASS_COLORS[lab]
                )
            
            if j == 0:
                label_text = r"Class $\mathcal{" + lab + r"}$"
                ax.text(
                    0.95, 0.1, label_text,
                    transform=ax.transAxes,
                    va="bottom", ha="right",
                    bbox=dict(
                        boxstyle="square,pad=0.2",
                        facecolor="white",
                        alpha=0.9,
                        linewidth=0.5,
                    ),
                )
            
            ax.set_ylim(y_lim_bottom, y_lim_top)
            ax.set_xlim(0, X.shape[1])
            
            if j > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                ax.yaxis.set_major_formatter(formatter)
    
    fig.text(
        0.03, 0.5, r"Amplitude [a.u.]", va="center", rotation="vertical", fontsize=9
    )
    fig.text(0.5, 0.03, r"Time $t$ [samples]", ha="center", fontsize=9)
    fig.suptitle(title, fontweight="bold", y=0.95)
    
    plt.subplots_adjust(left=0.1, wspace=0.1)
    
    fn = os.path.join(OUT_DIR, out_name)
    plt.savefig(fn.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn.replace('.png', '.pdf')}")

def plot_class_balance(y_original, filename="class_balance.pdf"):
    original_counts = Counter(y_original)
    labels = CLASS_ORDER
    x = np.arange(len(labels))
    width = 0.55

    n_count = original_counts["N"]
    ratios = [0.25, 0.50, 0.75, 1.00]

    # Same hue, different shades
    aug_colors = ["#CFE0F5", "#9FC2EA", "#6FA3DE", "#3F85D3"]

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 1.15, SINGLE_COLUMN * 0.95))

    # Original bars
    orig_vals = [original_counts[l] for l in labels]
    rects_orig = ax.bar(
        x,
        orig_vals,
        width,
        label="Original",
        color="#B0B0B0",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.95,
        zorder=3,
    )

    # Stack only on minority classes
    bottoms = np.array(orig_vals, dtype=float)
    prev_targets = {lab: original_counts[lab] for lab in labels}

    for ratio, color in zip(ratios, aug_colors):
        layer_vals = []
        for lab in labels:
            if lab == "N":
                layer_vals.append(0)
                continue

            target_total = max(original_counts[lab], int(round(n_count * ratio)))
            increment = max(0, target_total - prev_targets[lab])
            layer_vals.append(increment)
            prev_targets[lab] = target_total

        layer_vals = np.array(layer_vals, dtype=float)

        ax.bar(
            x,
            layer_vals,
            width,
            bottom=bottoms,
            label=rf"Added to ${ratio:.2f}\times N$",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch="////",
            zorder=3,
        )

        bottoms += layer_vals

    ax.set_yscale("log")
    ax.set_ylim(bottom=1, top=max(bottoms) * 6)

    # Put original labels higher inside the gray part
    for rect, val in zip(rects_orig, orig_vals):
        if val > 0:
            x_text = rect.get_x() + rect.get_width() / 2

            # higher placement inside the gray bar in log-space
            y_text = max(1.3, val ** 0.70)

            ax.text(
                x_text,
                y_text,
                f"{int(val)}",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
                zorder=5,
            )

    # Final totals on top of full stacks
    for xi, total in zip(x, bottoms):
        if total > 0:
            ax.annotate(
                f"{int(total)}",
                xy=(xi, total),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
            )

    ax.set_ylabel("Count (Log Scale)")
    ax.set_title("Train Class Distribution", fontweight="bold")
    ax.set_xticks(x)
    cal_labels = [rf"$\mathcal{{{l}}}$" for l in labels]
    ax.set_xticklabels(cal_labels, fontsize=9)

    sns.despine(ax=ax)
    ax.grid(axis="y", linestyle=":", alpha=0.35, which="both", zorder=0)

    ax.legend(
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.01),
    )

    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(
        fn.replace(".png", ".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()
    print(f"Saved: {fn.replace('.png', '.pdf')}")

def plot_qlr_shift(lab, mu_z, std_z, vae, latent_ddpm, qlr,
                   X_train, y_train, n_samples=300, seed=42):
    idx_r = np.where(y_train == lab)[0]
    if len(idx_r) < 50:
        return

    n_real = min(len(idx_r), 1500)
    X_real_samples = X_train[np.random.choice(idx_r, size=n_real, replace=False)]

    z_real_list = []
    bs = 256
    for i in range(0, len(X_real_samples), bs):
        batch = X_real_samples[i : i + bs]
        zb = encode_mu(vae, batch, [lab2idx[lab]] * len(batch))
        z_real_list.append(zb)
    z_real = np.vstack(z_real_list)

    c_labels = torch.full((n_samples,), lab2idx[lab], device=device, dtype=torch.long)
    with torch.no_grad():
        z_norm = latent_ddpm.sample(n_samples, c_labels).cpu().numpy()
        z_ddpm_raw = z_norm * std_z + mu_z
        z_qlr_raw = qlr.refine(lab, z_ddpm_raw)

    pca = PCA(n_components=2, random_state=seed)
    pca.fit(z_real)
    p_real = pca.transform(z_real)
    p_ddpm = pca.transform(z_ddpm_raw)
    p_qlr = pca.transform(z_qlr_raw)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))

    base_color = CLASS_COLORS[lab]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["#FFFFFF", base_color]
    )

    sns.kdeplot(
        x=p_real[:, 0],
        y=p_real[:, 1],
        fill=True,
        thresh=0.05,
        levels=100,
        cmap=cmap,
        alpha=0.35,
        ax=ax,
    )

    sns.kdeplot(
        x=p_real[:, 0],
        y=p_real[:, 1],
        levels=5,
        color=base_color,
        linewidths=0.3,
        alpha=0.3,
        ax=ax,
    )

    limit = min(50, n_samples)
    for i in range(limit):
        ax.plot(
            [p_ddpm[i, 0], p_qlr[i, 0]],
            [p_ddpm[i, 1], p_qlr[i, 1]],
            color="black",
            alpha=0.2,
            linewidth=0.6,
            zorder=1,
        )

    ax.scatter(
        p_ddpm[:limit, 0],
        p_ddpm[:limit, 1],
        s=15,
        facecolors="white",
        edgecolors="gray",
        linewidth=0.6,
        label=r"Initial latent $z_{ddpm}$",
        zorder=2,
        alpha=0.7,
    )

    ax.scatter(
        p_qlr[:limit, 0],
        p_qlr[:limit, 1],
        s=15,
        c="#222222",
        edgecolors="white",
        linewidth=0.3,
        label=r"Quantum refined $z_{qlr}$",
        zorder=3,
        alpha=1.0,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(rf"Latent Shift: Class $\mathcal{{{lab}}}$", fontweight="bold")

    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=False,
    )
    ax.grid(True, linestyle=":", alpha=0.3, color="gray")

    fn = os.path.join(OUT_DIR, f"qlr_shift_{lab}.pdf")
    plt.savefig(fn, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print(f"Saved: {fn}")

def plot_latent_space_cvae_ddpm(Z_train_mu, Zy, mu_z, std_z, latent_ddpm,
                                n_samples=150, seed=42,
                                filename="latent_space_cvae_ddpm.pdf"):
    minority_classes = ["S", "V", "F"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 1.1),
                             squeeze=False)

    pca = PCA(n_components=2, random_state=seed)
    pca.fit(Z_train_mu)
    ev = pca.explained_variance_ratio_
    print(f"[Latent PCA] PC1: {ev[0]*100:.1f}%  PC2: {ev[1]*100:.1f}%  "
          f"Cumulative: {sum(ev)*100:.1f}%")

    for col_idx, lab in enumerate(minority_classes):
        ax = axes[0, col_idx]
        lab_idx = lab2idx[lab]
        base_color = CLASS_COLORS[lab]

        # Real latents for this class
        real_mask = Zy == lab_idx
        Z_real_class = Z_train_mu[real_mask]
        P_real = pca.transform(Z_real_class)

        # DDPM samples for this class
        n_per = min(n_samples, len(Z_real_class))
        with torch.no_grad():
            c_labels = torch.full((n_per,), lab_idx, device=device, dtype=torch.long)
            z_norm = latent_ddpm.sample(n_per, c_labels).cpu().numpy()
            z_raw = z_norm * std_z + mu_z
        P_ddpm = pca.transform(z_raw)

        # KDE of real distribution
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "cls_cmap", ["#FFFFFF", base_color]
        )
        sns.kdeplot(x=P_real[:, 0], y=P_real[:, 1],
                    fill=True, thresh=0.05, levels=8,
                    cmap=cmap, alpha=0.5, ax=ax)
        sns.kdeplot(x=P_real[:, 0], y=P_real[:, 1],
                    levels=4, color=base_color,
                    linewidths=0.8, alpha=0.6, ax=ax)

        # DDPM samples as scatter
        ax.scatter(P_ddpm[:, 0], P_ddpm[:, 1],
                   s=10, color="#333333", alpha=0.25,
                   label=r"$z_{ddpm}$", zorder=3, rasterized=True)

        ax.set_title(rf"Class $\mathcal{{{lab}}}$", fontweight="bold", fontsize=9)
        ax.set_xlabel(rf"PC1", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(rf"PC2", fontsize=8)
        else:
            ax.set_yticklabels([])

        # Centroid drift annotation
        drift = np.sqrt(
            (P_real[:, 0].mean() - P_ddpm[:, 0].mean())**2 +
            (P_real[:, 1].mean() - P_ddpm[:, 1].mean())**2
        )
        ax.text(0.04, 0.04, rf"drift$={drift:.2f}$",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                          alpha=0.9, linewidth=0.5))

        # Add panel labels
        ax.text(-0.12, 1.05, ["A", "B", "C"][col_idx],
                transform=ax.transAxes, size=11, weight="bold")

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=5, label=r'DDPM $\hat{z}$'),
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.4,
                       label='Real KDE')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=8, frameon=True,
               fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

def plot_avg_morphology(X_train, y_train, synthetic_banks, lab):
    methods = ["cVAE", "DDPM", "DDPM+QLR"]

    idx_r = np.where(y_train == lab)[0]
    X_r = X_train[idx_r]
    mu_r = X_r.mean(axis=0)
    std_r = X_r.std(axis=0)
    t = np.arange(len(mu_r))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN * 1.3, SINGLE_COLUMN * 1.1),
                             sharey=True, sharex=True)

    print(f"\n[Morphology] Class {lab}  ({len(idx_r)} real beats)")
    print(f"  {'Method':<12} {'RMSD':>8} {'Max|Δ|':>10} {'CosSim':>8}")
    print(f"  {'-'*42}")

    base_color = CLASS_COLORS[lab]

    for col_idx, method in enumerate(methods):
        ax = axes[col_idx]
        synth = synthetic_banks.get(method, {}).get(lab, None)

        if synth is None or len(synth) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_title(method, fontweight="bold", fontsize=9)
            print(f"  {method:<12} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            continue

        mu_s  = synth.mean(axis=0)
        std_s = synth.std(axis=0)
        diff  = mu_s - mu_r

        rms_diff = np.sqrt(np.mean(diff ** 2))
        max_diff = np.max(np.abs(diff))
        cos_sim  = float(np.dot(mu_r, mu_s) /
                         (np.linalg.norm(mu_r) * np.linalg.norm(mu_s) + 1e-8))

        print(f"  {method:<12} {rms_diff:>8.4f} {max_diff:>10.4f} {cos_sim:>8.4f}")

        ax.fill_between(t, mu_r - std_r, mu_r + std_r,
                        color=base_color, alpha=0.2,
                        label=r"Real $\pm 1\sigma$")
        ax.plot(t, mu_r, color=base_color, linewidth=1.4,
                label=r"Real $\mu$")

        ax.fill_between(t, mu_s - std_s, mu_s + std_s,
                        color="gray", alpha=0.15,
                        label=r"Synth $\pm 1\sigma$")
        ax.plot(t, mu_s, color="black", linewidth=1.4, linestyle="--",
                label=r"Synth $\mu$")

        ax.text(0.97, 0.04,
                rf"RMSD$={rms_diff:.4f}$" + "\n" + rf"Max$|\Delta|={max_diff:.4f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                          alpha=0.9, linewidth=0.4))

        ax.set_title(method, fontweight="bold", fontsize=9)
        ax.set_xlabel(r"Time $t$ [samples]", fontsize=8)
        ax.text(-0.08, 1.05, ["A", "B", "C"][col_idx],
                transform=ax.transAxes, size=11, weight="bold")

        if col_idx == 0:
            ax.set_ylabel(r"Amplitude [a.u.]", fontsize=8)

        ax.legend(fontsize=6, loc="upper right", frameon=True,
                  fancybox=False, edgecolor="black")

    print(f"  {'-'*42}")

    fig.suptitle(rf"Morphology Comparison: Class $\mathcal{{{lab}}}$",
                 fontweight="bold", y=1.02)

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, f"morphology_{lab}.pdf")
    plt.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

def plot_tsne_quality(X_real, y_real, X_synth, y_synth, lab, title, filename):
    idx_r = np.where(y_real == lab)[0]
    X_r = X_real[idx_r]

    if len(X_r) > 500:
        X_r = X_r[np.random.choice(len(X_r), 500, replace=False)]
    if len(X_synth) > 500:
        X_synth = X_synth[np.random.choice(len(X_synth), 500, replace=False)]

    X_comb = np.vstack([X_r, X_synth])
    sources = np.concatenate([np.zeros(len(X_r)), np.ones(len(X_synth))])

    n_total = len(X_comb)
    if n_total < 5:
        return
    safe_perplexity = min(30, n_total - 1)

    X_pca = PCA(n_components=min(50, n_total - 1)).fit_transform(
        X_comb.reshape(n_total, -1)
    )
    X_tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        perplexity=safe_perplexity,
        learning_rate=200,
    ).fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.85))

    ax.scatter(
        X_tsne[sources == 1, 0],
        X_tsne[sources == 1, 1],
        c="gray",
        alpha=0.4,
        label=r"Synthetic Data $\mathcal{D}_S$",
        s=25,
        marker="^",
        edgecolor="white",
        linewidth=0.3,
        zorder=1,
    )

    ax.scatter(
        X_tsne[sources == 0, 0],
        X_tsne[sources == 0, 1],
        c=CLASS_COLORS[lab],
        alpha=0.8,
        label=r"Real Data $\mathcal{D}_R$",
        s=25,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"t-SNE Dimension 1")
    ax.set_ylabel(r"t-SNE Dimension 2")

    ax.legend(
        loc="upper right",
        fontsize=7,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.5,
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle=":", alpha=0.5)

    count_text = rf"$N_{{Real}}={len(X_r)}$" + "\n" + rf"$N_{{Synth}}={len(X_synth)}$"

    box_props = dict(
        boxstyle="square,pad=0.3",
        facecolor="white",
        alpha=0.95,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.text(
        0.95,
        0.05,
        count_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        bbox=box_props,
    )

    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn.replace('.png', '.pdf')}")

def plot_mmd_results(df_mmd):
    if df_mmd.empty:
        return

    method_colors = {
        "cVAE": "#4C72B0",
        "DDPM": "#DD8452",
        "DDPM+QLR": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.8))

    x = np.arange(len(df_mmd["Class"].unique()))
    width = 0.25
    methods = df_mmd["Method"].unique()

    for i, method in enumerate(methods):
        method_data = df_mmd[df_mmd["Method"] == method]
        values = method_data["MMD Mean"].values
        errors = method_data["MMD Std"].values

        col = method_colors.get(method, "gray")

        ax.bar(
            x + (i - len(methods) / 2 + 0.5) * width,
            values,
            width,
            yerr=errors,
            label=method,
            color=col,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            error_kw={"linewidth": 1},
        )

    ax.set_xlabel("ECG Class", fontweight="bold")
    ax.set_ylabel(r"MMD Distance ($D_{MMD}$)")
    ax.set_xticks(x)
    cal_classes = [rf"$\mathcal{{{c}}}$" for c in df_mmd["Class"].unique()]
    ax.set_xticklabels(cal_classes)

    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fn = os.path.join(OUT_DIR, "mmd_results.pdf")
    plt.savefig(fn, format="pdf")
    plt.close()
    print(f"Saved: {fn}")

def plot_diversity_fidelity(df_summary):
    if df_summary.empty:
        return

    method_colors = {
        "cVAE": "#4C72B0",
        "DDPM": "#DD8452",
        "DDPM+QLR": "#55A868",
    }
    
    desired_methods = ["cVAE", "DDPM", "DDPM+QLR"]
    methods = [m for m in desired_methods if m in df_summary["Method"].unique()]
    
    classes = sorted(df_summary["Class"].unique(), key=lambda x: CLASS_ORDER.index(x) if x in CLASS_ORDER else x)
    
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 0.75))
    
    metrics_config = [
        {
            "col_mean": "Diversity Mean",
            "col_std": "Diversity Std",
            "title": "Diversity (Euclidean Dist.)",
            "ylabel": r"Diversity ($\uparrow$)",
            "ax": axes[0]
        },
        {
            "col_mean": "Fidelity Mean",
            "col_std": "Fidelity Std",
            "title": "Fidelity (Cosine Dist.)",
            "ylabel": r"Fidelity ($\downarrow$)",
            "ax": axes[1]
        }
    ]
    
    x = np.arange(len(classes))
    width = 0.25
    total_width = width * len(methods)
    
    for config in metrics_config:
        ax = config["ax"]
        
        for i, method in enumerate(methods):
            method_data = df_summary[df_summary["Method"] == method]
            
            means = []
            stds = []
            for cls in classes:
                row = method_data[method_data["Class"] == cls]
                if not row.empty:
                    means.append(row[config["col_mean"]].values[0])
                    stds.append(row[config["col_std"]].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - len(methods) / 2 + 0.5) * width
            
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=method if config == metrics_config[0] else None,
                color=method_colors.get(method, "gray"),
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                error_kw={"linewidth": 1},
            )
            
        ax.set_xticks(x)
        cal_classes = [rf"$\mathcal{{{c}}}$" for c in classes]
        ax.set_xticklabels(cal_classes)
        ax.set_xlabel("ECG Class", fontweight="bold")
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"], fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=3, 
        frameon=True, 
        edgecolor="black", 
        fancybox=False, 
        fontsize=8
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    
    fn = os.path.join(OUT_DIR, "diversity_fidelity_comparison.pdf")
    plt.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

def plot_generative_training_dynamics(
    vae_train, vae_val, ddpm_train, ddpm_val, filename="training_dynamics.pdf"
):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 0.9))

    ax1 = axes[0]
    ep_vae = range(1, len(vae_train) + 1)
    ax1.plot(ep_vae, vae_train, color="#0072B2", linewidth=1.5, label="Train")
    if vae_val:
        ax1.plot(ep_vae, vae_val, color="#0072B2", linewidth=1.5,
                 linestyle="--", alpha=0.75, label="Validation")
    ax1.set_title(r"cVAE Convergence", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"Loss ($\mathcal{L}_{\mathrm{ELBO}}$)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)
    ax1.text(-0.15, 1.05, "A", transform=ax1.transAxes, size=12, weight="bold")

    ax2 = axes[1]
    ep_ddpm = range(1, len(ddpm_train) + 1)

    ax2.plot(ep_ddpm, ddpm_train, color="#D55E00", alpha=0.20,
             linewidth=0.8, label="_nolegend_")
    window = max(5, len(ddpm_train) // 20)
    if len(ddpm_train) > window:
        smoothed = np.convolve(ddpm_train, np.ones(window) / window, mode="valid")
        x_sm = np.arange(window // 2, len(smoothed) + window // 2) + 1
        ax2.plot(x_sm, smoothed, color="#D55E00", linewidth=1.5, label="Train (smoothed)")

    if ddpm_val:
        ax2.plot(ep_ddpm, ddpm_val, color="#CC79A7", linewidth=1.5,
                 linestyle="--", label="Validation")

    ax2.set_title(r"Latent DDPM Convergence", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"MSE Loss")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    ax2.text(-0.15, 1.05, "B", transform=ax2.transAxes, size=12, weight="bold")

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

def plot_cm(y_true, y_pred, title, filename):
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_ORDER)))
    
    cm_sum = cm_raw.sum(axis=1)[:, np.newaxis]
    cm_norm = cm_raw.astype("float") / (cm_sum + 1e-8)
    
    cal_order = [rf"$\mathcal{{{c}}}$" for c in CLASS_ORDER]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.9))
    
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=cal_order,
        yticklabels=cal_order,
        square=True,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        ax=ax,
    )
    
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            pct = cm_norm[i, j] * 100
            count = cm_raw[i, j]
            
            text_color = "white" if pct > 50 else "black"
            
            ax.text(
                j + 0.5, i + 0.45, f"{pct:.1f}%",
                ha="center", va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                j + 0.5, i + 0.75, f"({count})",
                ha="center", va="center",
                color=text_color,
                fontsize=6,
                alpha=0.8,
            )
    
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    
    plt.yticks(rotation=0)
    
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(
        fn.replace(".png", ".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.05
    )
    plt.close()
    print(f"Saved: {fn.replace('.png', '.pdf')}")

def plot_comprehensive_results(df_results):
    set_publication_style(use_latex=has_latex)
    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.72), constrained_layout=True
    )
    
    default_palette = sns.color_palette("deep", 10)
    method_colors = {
        "cVAE": default_palette[0],
        "DDPM": default_palette[1],
        "DDPM+QLR": default_palette[2],
        "SMOTE": default_palette[3],
    }
    
    df_aug = df_results[df_results["Method"] != "Baseline"]
    methods_list = ["SMOTE", "cVAE", "DDPM", "DDPM+QLR"]
    ratios = np.sort(df_aug["Ratio"].unique())
    x = np.arange(len(ratios))
    bar_width = 0.8 / len(methods_list)
    
    # Macro F1 Score
    ax1 = axes[0, 0]
    for i, method in enumerate(methods_list):
        scores = [
            df_aug[(df_aug["Method"] == method) & (df_aug["Ratio"] == r)][
                "Macro_F1"
            ].iloc[0]
            for r in ratios
        ]
        offset = (i - (len(methods_list) - 1) / 2) * bar_width
        col = method_colors.get(method, default_palette[i % 10])
        ax1.bar(
            x + offset, scores, width=bar_width,
            label=method,
            color=col,
            edgecolor="black",
            linewidth=0.5,
        )
    
    if "Baseline" in df_results["Method"].values:
        base_val = df_results[df_results["Method"] == "Baseline"]["Macro_F1"].iloc[0]
        ax1.axhline(
            base_val, color="gray", linestyle="--", linewidth=1.2, label="Baseline"
        )
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\rho={r}$" for r in ratios])
    ax1.set_ylabel("Macro F1 Score")
    ax1.set_ylim(0.75, 0.88)
    ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, size=12, weight="bold")
    
    ax1.legend(
        loc="upper right",
        fontsize=7,
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    
    # Per-Class Sensitivity
    ax2 = axes[0, 1]
    best_method = "DDPM+QLR"
    best_data = df_results[df_results["Method"] == best_method].sort_values("Ratio")
    metrics, class_labels = ["N_SEN", "S_SEN", "V_SEN", "F_SEN"], [r"$\mathcal{N}$", r"$\mathcal{S}$", r"$\mathcal{V}$", r"$\mathcal{F}$"]
    
    ratio_palette = sns.color_palette("Blues", n_colors=len(ratios))
    bar_width_b = 0.8 / len(ratios)
    
    for i, r in enumerate(ratios):
        vals = [best_data[best_data["Ratio"] == r][m].iloc[0] for m in metrics]
        offset = (i - (len(ratios) - 1) / 2) * bar_width_b
        ax2.bar(
            np.arange(4) + offset, vals, width=bar_width_b,
            label=rf"$\rho={r}$",
            color=ratio_palette[i],
            edgecolor="black",
            linewidth=0.5,
        )
    
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(class_labels)
    ax2.set_ylabel("Sensitivity (Recall)")
    ax2.set_title(f"Per-Class Sens. ({best_method})", fontsize=9)
    ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, size=12, weight="bold")
    ax2.legend(fontsize=7, title=r"Ratio $\rho$", loc="lower right")
    
    # Overall Accuracy
    ax3 = axes[1, 0]
    for i, method in enumerate(methods_list):
        accs = [
            df_aug[(df_aug["Method"] == method) & (df_aug["Ratio"] == r)]["ACC"].iloc[0]
            for r in ratios
        ]
        offset = (i - (len(methods_list) - 1) / 2) * bar_width
        col = method_colors.get(method, default_palette[i % 10])
        ax3.bar(
            x + offset, accs, width=bar_width,
            color=col,
            edgecolor="black",
            linewidth=0.5,
        )
    
    if "Baseline" in df_results["Method"].values:
        base_acc = df_results[df_results["Method"] == "Baseline"]["ACC"].iloc[0]
        ax3.axhline(base_acc, color="gray", linestyle="--", linewidth=1.2)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"$\\rho={r}$" for r in ratios])
    ax3.set_ylabel("Overall Accuracy")
    ax3.set_xlabel(r"Augmentation Ratio $\rho$")
    ax3.set_ylim(0.85, 1.0)
    ax3.text(-0.12, 1.05, "C", transform=ax3.transAxes, size=12, weight="bold")
    
    # Relative Improvement
    ax4 = axes[1, 1]
    if "Baseline" in df_results["Method"].values:
        baseline_f1 = df_results[df_results["Method"] == "Baseline"]["Macro_F1"].iloc[0]
        imp_data, methods_d = [], []
        
        for m in methods_list:
            m_max = df_results[df_results["Method"] == m]["Macro_F1"].max()
            imp_data.append(((m_max - baseline_f1) / baseline_f1) * 100)
            methods_d.append(m)
        
        idx_sort = np.argsort(imp_data)
        imp_sorted, methods_sorted = (
            np.array(imp_data)[idx_sort],
            np.array(methods_d)[idx_sort],
        )
        
        bars = ax4.barh(
            np.arange(len(methods_sorted)), imp_sorted,
            color=[method_colors.get(m, default_palette[0]) for m in methods_sorted],
            edgecolor="black",
            height=0.6,
        )
        
        ax4.set_yticks(np.arange(len(methods_sorted)))
        ax4.set_yticklabels(methods_sorted)
        ax4.set_xlabel("Max Rel. Improv. [%]")
        
        outline = [path_effects.withStroke(linewidth=1.5, foreground="black")]
        
        for rect in bars:
            val = rect.get_width()
            txt = ax4.text(
                val - 0.05,
                rect.get_y() + rect.get_height() / 2,
                f"{val:+.1f}%",
                va="center",
                ha="right",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            txt.set_path_effects(outline)
    
    ax4.text(-0.12, 1.05, "D", transform=ax4.transAxes, size=12, weight="bold")
    sns.despine(fig=fig)
    
    fn = os.path.join(OUT_DIR, "comprehensive_results.pdf")
    plt.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")

def plot_seed_variance(df_master):
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, SINGLE_COLUMN))

    for method in ["SMOTE", "cVAE", "DDPM", "DDPM+QLR"]:
        style = METHOD_STYLES.get(method, {})
        method_data = df_master[df_master["Method"] == method]
        grouped = method_data.groupby("Ratio")["Macro_F1"].agg(["mean", "std"])

        ax.errorbar(
            grouped.index, grouped["mean"], yerr=grouped["std"],
            label=method,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            capsize=3, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel(r"Augmentation Ratio $\rho$")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Multi-Seed Variance Analysis", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    fn = os.path.join(OUT_DIR, "seed_variance.pdf")
    plt.savefig(fn, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {fn}")

def plot_ablation_study(ablation_results, filename="ablation_study.pdf"):
    """Two-panel figure: loss curves (left) + downstream V-class F1 bar chart (right)."""
    has_downstream = "downstream" in ablation_results
    ncols = 2 if has_downstream else 1
    fig, axes = plt.subplots(1, ncols,
                             figsize=(DOUBLE_COLUMN if has_downstream else SINGLE_COLUMN,
                                      SINGLE_COLUMN * 0.85))
    if ncols == 1:
        axes = [axes]
    ax = axes[0]
    window = 5

    mlp_colors = {"MLP-S": "#D55E00", "MLP-M": "#CC79A7", "MLP-L": "#56B4E9"}
    mlp_ls     = {"MLP-S": "--",      "MLP-M": "-.",      "MLP-L": ":"}

    plot_series = [("Quantum_val", "#0072B2", "-", r"QLR (689 params)")]
    for name in ABLATION_MLP_CONFIGS:
        key = f"{name}_val"
        if key in ablation_results:
            n_params = {"MLP-S": "~300", "MLP-M": "~1100", "MLP-L": "~2100"}.get(name, "?")
            plot_series.append((key, mlp_colors[name], mlp_ls[name], rf"{name} ({n_params} params)"))

    for key, color, ls, label in plot_series:
        raw = ablation_results.get(key, [])
        raw_clean = [v for v in raw if not (isinstance(v, float) and math.isnan(v))]
        ax.plot(raw, color=color, linestyle=ls, linewidth=1.0, alpha=0.25, label=label)
        if len(raw_clean) > window:
            smooth = np.convolve(raw_clean, np.ones(window) / window, mode="valid")
            x_s = range(window // 2, len(smooth) + window // 2)
            ax.plot(x_s, smooth, color=color, linestyle=ls, linewidth=2.0, alpha=1.0,
                    label="_nolegend_")

    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"Validation Loss $\mathcal{L}_{total}$")
    ax.set_title(r"Ablation: Latent Distribution Loss", fontweight="bold", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ---- right panel: downstream classification F1 bar chart ---------------
    if has_downstream:
        ax2 = axes[1]
        downstream = ablation_results["downstream"]
        # Determine the minority class from results keys
        f1_key = next((k for k in next(iter(downstream.values())) if k.endswith("_F1")
                       and k != "F1"), "V_F1")
        lab_tag = f1_key.replace("_F1", "")
        lab_tag_math = rf"$\mathcal{{{lab_tag}}}$"

        bar_colors = {
            "DDPM": "#999999",
            "QLR":              "#0072B2",
            "MLP-S":            "#D55E00",
            "MLP-M":            "#CC79A7",
            "MLP-L":            "#56B4E9",
        }
        names = list(downstream.keys())
        f1_vals   = [downstream[n].get(f1_key,  0.0) for n in names]
        macro_vals = [downstream[n].get("F1",    0.0) for n in names]

        x = np.arange(len(names))
        w = 0.35
        colors = [bar_colors.get(n, "#444444") for n in names]

        b1 = ax2.barh(x + w/2, f1_vals,   height=w, color=colors, alpha=0.9,
                      label=f"{lab_tag_math}-class F1")
        b2 = ax2.barh(x - w/2, macro_vals, height=w, color=colors, alpha=0.45,
                      hatch="///", label="Macro F1")

        ax2.set_yticks(x)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("F1 Score", fontsize=8)
        ax2.set_title(f"Ablation: Downstream Classification\n({lab_tag_math}-class vs Macro F1)",
                      fontweight="bold", fontsize=9)
        ax2.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="black")
        ax2.grid(axis="x", linestyle=":", alpha=0.5)
        # Pad past 1.0 so a bar/label near the F1 ceiling (e.g. QLR's
        # V-class score) always has room before the right spine; ticks are
        # pinned back to the normal 0-1.0 range so the margin stays invisible.
        ax2.set_xlim(0, 1.08)
        ax2.set_xticks(np.arange(0, 1.01, 0.2))

        # Annotate bars with value
        for bar in list(b1) + list(b2):
            w_ = bar.get_width()
            ax2.text(w_ + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{w_:.3f}", va="center", ha="left", fontsize=6.5)

    if "mmd_results" in ablation_results:
        print("\n[Ablation] MMD reduction summary:")
        for k, v in ablation_results["mmd_results"].items():
            print(f"  {k:12s}: {v:.6f}")

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {fn}")

# =============================================================================
# MAIN EXPERIMENTAL LOOP
# =============================================================================

def run_single_seed_experiment(current_seed, fixed_data):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT RUN: SEED {current_seed}")
    print(f"{'='*60}\n")
    
    set_seed(current_seed)
    
    X_train = fixed_data["X_train"]
    y_train = fixed_data["y_train"]
    X_val   = fixed_data["X_val"]
    y_val   = fixed_data["y_val"]
    X_test  = fixed_data["X_test"]
    y_test  = fixed_data["y_test"]
    input_len = fixed_data["input_len"]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    
    y_train_idx = torch.tensor([lab2idx[l] for l in y_train], dtype=torch.long)
    y_val_idx   = torch.tensor([lab2idx[l] for l in y_val], dtype=torch.long)
    y_test_idx  = torch.tensor([lab2idx[l] for l in y_test], dtype=torch.long)
    
    # Weights and Samplers
    cls_counts = np.bincount(y_train_idx.numpy(), minlength=cond_dim)
    cls_weights = 1.0 / np.maximum(cls_counts, 1)
    sample_weights = cls_weights[y_train_idx.numpy()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train_idx), replacement=True)

    # Loaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_idx), batch_size=BATCH_SIZE, 
                              sampler=sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_idx), batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_idx), batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=NUM_WORKERS)
    
    # Train VAE
    print(f"[Seed {current_seed}] Training VAE...")
    vae = CVAE(input_len=input_len, cond_dim=cond_dim, latent_dim=LATENT_DIM).to(device)
    _t0 = time.time()
    train_losses, val_losses = train_vae(vae, train_loader, val_loader)
    timing_log[f"seed{current_seed}_vae"] = time.time() - _t0
    print(f"  VAE training time: {timing_log[f'seed{current_seed}_vae']:.1f}s")
    torch.save(vae.state_dict(), os.path.join(OUT_DIR, f"vae_seed{current_seed}.pth"))


    torch.cuda.empty_cache()
    
    # Train DDPM
    print(f"[Seed {current_seed}] Encoding latents...")
    Z_train_mu = encode_mu(vae, X_train, [lab2idx[l] for l in y_train])

    mu_z = np.mean(Z_train_mu, axis=0)
    std_z = np.std(Z_train_mu, axis=0) + 1e-8
    Z_train_norm = (Z_train_mu - mu_z) / std_z
    
    Z_train_t = torch.tensor(Z_train_norm, dtype=torch.float32)
    y_train_lat_t = torch.tensor([lab2idx[l] for l in y_train], dtype=torch.long)

    mu_z_t = torch.tensor(mu_z, device=device, dtype=torch.float32)
    std_z_t = torch.tensor(std_z, device=device, dtype=torch.float32)
    
    Zy = np.array([lab2idx[l] for l in y_train])

    latent_ds = TensorDataset(Z_train_t.unsqueeze(1), y_train_lat_t)
    latent_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    Z_val_mu = encode_mu(vae, X_val, [lab2idx[l] for l in y_val])
    Z_val_norm = (Z_val_mu - mu_z) / std_z
    Z_val_t = torch.tensor(Z_val_norm, dtype=torch.float32)
    y_val_lat_t = torch.tensor([lab2idx[l] for l in y_val], dtype=torch.long)

    latent_val_ds = TensorDataset(Z_val_t.unsqueeze(1), y_val_lat_t)

    print(f"[Seed {current_seed}] Training DDPM...")
    ddpm_net = LatentMLPDenoiser(cond_dim=cond_dim, latent_dim=LATENT_DIM, width=256)
    latent_ddpm = LatentDDPM(ddpm_net, latent_dim=LATENT_DIM, timesteps=DDPM_TIMESTEPS)

    _t0 = time.time()
    latent_ddpm, ddpm_train_history, ddpm_val_history = train_latent_ddpm(
        latent_ddpm, latent_ds, epochs=DDPM_EPOCHS, bs=BATCH_SIZE,
        lr=DDPM_LR, sampler=latent_sampler, val_ds=latent_val_ds
    )
    timing_log[f"seed{current_seed}_ddpm"] = time.time() - _t0
    print(f"  DDPM training time: {timing_log[f'seed{current_seed}_ddpm']:.1f}s")
    torch.save(latent_ddpm.model.state_dict(), os.path.join(OUT_DIR, f"ddpm_seed{current_seed}.pth"))

    print("\n=== DDPM Quality Check ===")

    with torch.no_grad():
        for lab in ["S", "V", "F"]:
            idx_real = np.where(Zy == lab2idx[lab])[0]
            if len(idx_real) == 0:
                print(f"Class {lab}: No real training samples — skipping quality check")
                continue
            c_labels = torch.full((256,), lab2idx[lab], device=device)
            z_norm = latent_ddpm.sample(256, c_labels)
            z_vae = z_norm * std_z_t + mu_z_t
            sel_idx = idx_real[:256] if len(idx_real) >= 256 else np.random.choice(idx_real, 256, replace=True)
            z_real = torch.tensor(Z_train_mu[sel_idx], device=device, dtype=torch.float32)
            dists = torch.cdist(z_vae, z_real, p=2)
            avg_dist = dists.min(dim=1)[0].mean().item()
            print(f"Class {lab}: DDPM samples avg distance to real = {avg_dist:.4f}")

            for other in CLASS_ORDER:
                if other == lab:
                    continue
                idx_other = np.where(Zy == lab2idx[other])[0]
                if len(idx_other) == 0:
                    continue
                sel_other = (
                    idx_other[:256] if len(idx_other) >= 256
                    else np.random.choice(idx_other, 256, replace=True)
                )
                z_other = torch.tensor(Z_train_mu[sel_other], device=device, dtype=torch.float32)
                cross_dist = torch.cdist(z_vae, z_other, p=2).min(dim=1)[0].mean().item()
                print(f"  Class {lab} DDPM samples avg distance to real_{other} = {cross_dist:.4f}")

    torch.cuda.empty_cache()
    
    # Train QLR
    print(f"[Seed {current_seed}] Training QLR...")

    qlr = QuantumRefiner(
        n_layers=6, alpha=QLR_ALPHA, lr=QLR_LR, minority_only=True
    )

    _t0 = time.time()
    qlr.fit(
        Z_train_mu,
        Zy,
        latent_ddpm,
        mu_z_t, std_z_t,
        iters=QLR_EPOCHS, bs=128
    )
    timing_log[f"seed{current_seed}_qlr"] = time.time() - _t0
    print(f"  QLR training time: {timing_log[f'seed{current_seed}_qlr']:.1f}s")
    torch.save(qlr.state_dict(), os.path.join(OUT_DIR, f"qlr_seed{current_seed}.pth"))

    print("\n=== QLR Refinement Check ===")
    
    with torch.no_grad():
        for lab in ["S", "V", "F"]:
            if np.sum(Zy == lab2idx[lab]) == 0:
                print(f"Class {lab}: No real training samples — skipping QLR check")
                continue
            c_labels = torch.full((256,), lab2idx[lab], device=device)

            z_norm_ddpm = latent_ddpm.sample(256, c_labels)
            z_vae_ddpm = (z_norm_ddpm * std_z_t + mu_z_t).cpu().numpy()
            
            z_vae_qlr = qlr.refine(lab, z_vae_ddpm)
            
            movement = np.linalg.norm(z_vae_qlr - z_vae_ddpm, axis=1).mean()
            
            print(f"Class {lab}: QLR movement distance = {movement:.4f}")

    torch.cuda.empty_cache()
    
    if current_seed == GLOBAL_SEEDS[0]:
        print(f"[Seed {current_seed}] Running ablation study...")
        
        ablation_class = "V"
        idx_ab = np.where(Zy == lab2idx[ablation_class])[0]
        
        if len(idx_ab) > 100:
            real_vae_data = Z_train_mu[idx_ab]
            ab_min_t, ab_max_t = get_robust_bounds(real_vae_data)
            
            ablation_results = run_ablation_comparison(
                ablation_class,
                real_vae_data,
                latent_ddpm,
                ab_min_t, ab_max_t,
                lambda z: from_unit_robust(z, ab_min_t, ab_max_t),
                mu_z_t, std_z_t,
                vae=vae,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test,
                n_iters=80,
                bs=128,
            )
            
            plot_ablation_study(ablation_results, filename="ablation_study.pdf")
        else:
            print(f"[Warning] Not enough samples for ablation study on class {ablation_class}")

    print(f"[Seed {current_seed}] Generating synthetic data...")
    synthetic_banks = {"cVAE": {}, "DDPM": {}, "DDPM+QLR": {}}
    counts = Counter(y_train)
    maj_count = counts["N"]
    
    for mode in synthetic_banks.keys():
        for lab in CLASS_ORDER:
            if lab == "N":
                continue
            gap = max(0, maj_count - counts[lab])
            if gap > 0:
                generated_data = generate_data(
                    mode, lab, gap, vae, latent_ddpm, qlr, mu_z, std_z
                )
                synthetic_banks[mode][lab] = generated_data
    
    print(f"[Seed {current_seed}] Computing quality metrics...")
    quality_metrics = []

    for mode in ["cVAE", "DDPM", "DDPM+QLR"]:
        for lab in ["S", "V", "F"]:
            if lab not in synthetic_banks[mode] or len(synthetic_banks[mode][lab]) == 0:
                continue
                
            idx_real = np.where(y_train == lab)[0]
            X_real_class = X_train[idx_real]
            X_synth_class = synthetic_banks[mode][lab]
            
            mmd_score = mmd_gaussian(X_real_class, X_synth_class)
            diversity, fidelity = calculate_diversity_metrics(X_real_class, X_synth_class)
            cos_score = calculate_cosine_score(X_real_class, X_synth_class)
            
            quality_metrics.append({
                "Seed": current_seed,
                "Method": mode,
                "Class": lab,
                "MMD": mmd_score,
                "Diversity": diversity,
                "Fidelity": fidelity,
                "CosSim": cos_score
            })
            
    # =========================================================================
    # GENERATE PLOTS (Only for first seed)
    # =========================================================================

    if current_seed == GLOBAL_SEEDS[0]:
        print(f"\n[Seed {current_seed}] Generating publication figures...")
        
        # 1. Random beat examples
        plot_random_beats(X_train, y_train, "Training Data Samples", "train_beats.png")

        # 2. Training dynamics
        plot_generative_training_dynamics(
            train_losses, val_losses,
            ddpm_train_history, ddpm_val_history,
            filename="training_dynamics.pdf"
        )

        # 3. QLR shift visualizations
        print("Generating QLR Shift plots...")
        for lab in ["S", "V", "F"]:
            plot_qlr_shift(lab, mu_z, std_z, vae, latent_ddpm, qlr,
                           X_train, y_train, n_samples=300, seed=current_seed)

        # 4. cVAE DDPM visualizations
        print("Generating cVAE-DDPM plot...")
        plot_latent_space_cvae_ddpm(
            Z_train_mu, Zy, mu_z, std_z, latent_ddpm,
            n_samples=150, seed=current_seed,
            filename="latent_space_cvae_ddpm.pdf"
        )
        
        # 5. Morphology and t-SNE
        print("Generating Morphology and t-SNE plots...")
        for lab in ["S", "V", "F"]:
            plot_avg_morphology(X_train, y_train, synthetic_banks, lab)

        for mode in ["cVAE", "DDPM", "DDPM+QLR"]:
            for lab in ["S", "V", "F"]:
                if lab in synthetic_banks[mode] and len(synthetic_banks[mode][lab]) > 50:
                    plot_tsne_quality(
                        X_train, y_train, synthetic_banks[mode][lab],
                        [lab] * len(synthetic_banks[mode][lab]), lab,
                        f"{mode} Distribution", f"tsne_{mode}_{lab}.pdf"
                    )

    # =========================================================================
    # CLASSIFICATION EXPERIMENTS
    # =========================================================================

    print(f"[Seed {current_seed}] Running classification experiments...")
    results = []
    
    # TEST MULTIPLE RATIOS TO SEE FULL PERFORMANCE CURVES
    method_ratios = {
        "SMOTE": [0.25, 0.5, 0.75, 1.0],
        "cVAE": [0.25, 0.5, 0.75, 1.0],
        "DDPM": [0.25, 0.5, 0.75, 1.0],
        "DDPM+QLR": [0.25, 0.5, 0.75, 1.0]
    }

    best_info = {}
    best_test_info = {}
    
    print(f"[Seed {current_seed}] Training Baseline...")
    base_model = MobileNetV2_1D(num_classes=cond_dim, input_channels=1).to(device)
    base_model, base_val_f1 = train_clf(base_model, train_loader, val_loader, epochs=CLF_EPOCHS)
    base_res = evaluate(base_model, test_loader)
    
    # SAVE BASELINE MODEL
    torch.save(base_model.state_dict(), os.path.join(OUT_DIR, f"model_Baseline_r0.0_seed{current_seed}.pth"))
    
    results.append({"Method": "Baseline", "Ratio": 0.0, "Seed": current_seed, **base_res["metrics"]})
    best_info["Baseline"] = {"ratio": 0.0, "val_f1": base_val_f1, "model": copy.deepcopy(base_model)}
    best_test_info["Baseline"] = {"ratio": 0.0, "test_f1": base_res["metrics"]["Macro_F1"], "y_true": base_res["y_true"], "y_pred": base_res["y_pred"]}

    for mode in ["SMOTE", "cVAE", "DDPM", "DDPM+QLR"]:
        ratios_to_test = method_ratios[mode]
        
        for r in ratios_to_test:
            print(f"[Seed {current_seed}] Training {mode} | Ratio {r}")
            
            X_aug_list = [X_train]
            y_aug_list = [y_train]
            
            if mode == "SMOTE":
                target_counts = {}
                for c in CLASS_ORDER:
                    if c == "N": continue
                    if int(maj_count * r) > counts[c]:
                        target_counts[lab2idx[c]] = int(maj_count * r)
                
                if not target_counts: continue
                sm = SMOTE(sampling_strategy=target_counts, k_neighbors=5, random_state=current_seed)
                x_res, y_res = sm.fit_resample(X_train.reshape(len(X_train), -1), [lab2idx[l] for l in y_train])
                X_aug, y_aug = x_res.reshape(-1, input_len), np.array([idx2lab[i] for i in y_res])
            else:
                for lab in CLASS_ORDER:
                    if lab == "N": continue
                    gap = max(0, maj_count - counts[lab])
                    n_to_gen = int(gap * r)

                    if n_to_gen > 0:
                        if len(synthetic_banks[mode][lab]) < n_to_gen:
                            more = generate_data(mode, lab, gap, vae, latent_ddpm, qlr, mu_z, std_z)
                            synthetic_banks[mode][lab] = np.vstack([synthetic_banks[mode][lab], more])
                        
                        idx = np.random.choice(len(synthetic_banks[mode][lab]), size=n_to_gen, replace=False)
                        X_aug_list.append(synthetic_banks[mode][lab][idx])
                        y_aug_list.append([lab] * n_to_gen)

                X_aug = np.concatenate(X_aug_list, axis=0)
                y_aug = np.concatenate(y_aug_list, axis=0)
            
            X_aug_t = torch.tensor(X_aug, dtype=torch.float32).unsqueeze(1)
            y_aug_idx = torch.tensor([lab2idx[l] for l in y_aug], dtype=torch.long)
            aug_loader = DataLoader(TensorDataset(X_aug_t, y_aug_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            
            clf_model = MobileNetV2_1D(num_classes=cond_dim, input_channels=1).to(device)
            clf_model, f1_val = train_clf(clf_model, aug_loader, val_loader, epochs=CLF_EPOCHS)
            
            # SAVE AUGMENTED MODEL
            torch.save(clf_model.state_dict(), os.path.join(OUT_DIR, f"model_{mode}_r{r}_seed{current_seed}.pth"))
            
            res = evaluate(clf_model, test_loader)
            results.append({"Method": mode, "Ratio": r, "Seed": current_seed, **res["metrics"]})

            test_f1 = res["metrics"]["Macro_F1"]

            if mode not in best_test_info or test_f1 > best_test_info[mode]["test_f1"]:
                best_test_info[mode] = {
                    "ratio": r,
                    "test_f1": test_f1,
                    "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                }
            
            if mode not in best_info or f1_val > best_info[mode]["val_f1"]:
                best_info[mode] = {
                    "ratio": r,
                    "val_f1": f1_val,
                    "model": copy.deepcopy(clf_model)
                }
            
            del clf_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Class balance and confusion matrices run ONCE after all modes complete
    if current_seed == GLOBAL_SEEDS[0]:
        plot_class_balance(y_train, filename="class_balance.pdf")

        print("Generating Confusion Matrices...")
        for method in ["Baseline", "SMOTE", "cVAE", "DDPM", "DDPM+QLR"]:
            if method in best_test_info:
                info = best_test_info[method]
                print(
                    f"{method:12s}: Ratio={info['ratio']:.2f}, "
                    f"Test F1={info['test_f1']:.4f}"
                )
                plot_cm(
                    info["y_true"],
                    info["y_pred"],
                    title=f"{method} (Best test ratio={info['ratio']}, Test F1={info['test_f1']:.3f})",
                    filename=f"cm_{method}_best_seed{current_seed}.pdf"
                )
        print(f"[Seed {current_seed}] Plotting complete!")
    
    return results, quality_metrics

# =============================================================================
# STATISTICAL REPORTING FUNCTION
# =============================================================================

def _ci95(values):
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    lo, hi = stats.t.interval(0.95, df=n - 1,
                               loc=np.mean(values),
                               scale=stats.sem(values))
    return lo, hi


def _cohens_d(a, b):
    """Cohen's d for paired comparison (pooled std)."""
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")


def report_scientific_summary(df):
    n_seeds = df["Seed"].nunique()
    print("\n" + "=" * 65)
    print(f" SCIENTIFIC SUMMARY: MEAN ± STD ({n_seeds} SEEDS) ")
    print("=" * 65)

    numeric_cols = ["Macro_F1", "ACC", "Macro_Recall"]
    per_class_f1_cols = [c for c in df.columns if c.endswith("_F1") and c != "Macro_F1"]

    # ------------------------------------------------------------------ aggregate
    summary = df.groupby(["Method", "Ratio"])[numeric_cols].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    # 95% CI column for Macro F1
    def _ci_str(grp):
        vals = grp["Macro_F1"].values
        lo, hi = _ci95(vals)
        return f"[{lo:.4f}, {hi:.4f}]"

    ci_df = df.groupby(["Method", "Ratio"]).apply(_ci_str).reset_index()
    ci_df.columns = ["Method", "Ratio", "Macro_F1_CI95"]
    summary = summary.merge(ci_df, on=["Method", "Ratio"])

    for col in numeric_cols:
        summary[f"{col}_Report"] = summary.apply(
            lambda x: f"{x[f'{col}_mean']:.4f} ± {x[f'{col}_std']:.4f}", axis=1
        )

    summary_path = os.path.join(OUT_DIR, "scientific_summary_report.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n>>> Detailed summary saved to: {summary_path}")

    # ------------------------------------------------------------------ baseline
    baseline_df = summary[summary["Method"] == "Baseline"]
    if not baseline_df.empty:
        row = baseline_df.iloc[0]
        print("\n" + "-" * 40)
        print("BASELINE PERFORMANCE (No Aug)")
        print("-" * 40)
        vals = df[df["Method"] == "Baseline"]["Macro_F1"].values
        lo, hi = _ci95(vals)
        print(f"Macro F1 : {row['Macro_F1_Report']}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"Accuracy : {row['ACC_Report']}")

    # ------------------------------------------------------------------ augmentation table
    aug_df = summary[summary["Method"] != "Baseline"]
    if not aug_df.empty:
        print("\n" + "-" * 65)
        print("AUGMENTATION PERFORMANCE (Macro F1 mean ± std  [95% CI])")
        print("-" * 65)
        pivot = aug_df.pivot(index="Method", columns="Ratio", values="Macro_F1_Report")
        with pd.option_context("display.max_rows", None, "display.max_columns", None,
                               "display.width", 1200):
            print(pivot)

    if per_class_f1_cols:
        print("\n" + "-" * 65)
        print("PER-CLASS F1 AT BEST RATIO (mean ± std across seeds)")
        print("-" * 65)
        best_rows = (
            df[df["Method"] != "Baseline"]
            .groupby(["Method", "Ratio"])["Macro_F1"]
            .mean()
            .reset_index()
            .loc[lambda d: d.groupby("Method")["Macro_F1"].idxmax()]
        )
        for _, brow in best_rows.iterrows():
            m, r = brow["Method"], brow["Ratio"]
            sub = df[(df["Method"] == m) & (df["Ratio"] == r)]
            parts = [f"{m} (ρ={r:.2f})"]
            for fc in per_class_f1_cols:
                if fc in sub.columns:
                    parts.append(f"  {fc}: {sub[fc].mean():.4f} ± {sub[fc].std():.4f}")
            print("\n".join(parts))

    print("\n" + "-" * 65)
    print(" STATISTICAL SIGNIFICANCE + EFFECT SIZES")
    print("-" * 65)

    try:
        comparison_ratio = 1.0
        qlr_scores  = df[(df["Method"] == "DDPM+QLR") & (df["Ratio"] == comparison_ratio)]["Macro_F1"].values
        ddpm_scores = df[(df["Method"] == "DDPM")     & (df["Ratio"] == comparison_ratio)]["Macro_F1"].values

        if len(qlr_scores) > 1 and len(ddpm_scores) > 1:
            t_stat, p_val = stats.ttest_rel(qlr_scores, ddpm_scores)
            d = _cohens_d(qlr_scores, ddpm_scores)
            lo_q, hi_q = _ci95(qlr_scores)
            lo_d, hi_d = _ci95(ddpm_scores)

            print(f"Comparison at Ratio {comparison_ratio} ({len(qlr_scores)} seeds)")
            print(f"DDPM     : {np.mean(ddpm_scores):.4f}  95% CI [{lo_d:.4f}, {hi_d:.4f}]")
            print(f"DDPM+QLR : {np.mean(qlr_scores):.4f}  95% CI [{lo_q:.4f}, {hi_q:.4f}]")
            print(f"Δ (QLR-DDPM) : {np.mean(qlr_scores) - np.mean(ddpm_scores):+.4f}")
            print(f"Cohen's d    : {d:.3f}")
            print(f"p-value (paired t) : {p_val:.5f}")
            if p_val < 0.05:
                print(">> STATISTICALLY SIGNIFICANT (p < 0.05)")
            else:
                print(">> Not significant at α=0.05")
        else:
            print("Insufficient data for t-test.")
    except Exception as e:
        print(f"Could not perform statistical analysis: {e}")

    if timing_log:
        print("\n" + "-" * 65)
        print(" TIMING SUMMARY")
        print("-" * 65)
        import platform, torch as _torch
        print(f"  Platform : {platform.node()}  {platform.processor()}")
        print(f"  Python   : {platform.python_version()}")
        print(f"  PyTorch  : {_torch.__version__}")
        print(f"  Device   : {device}")
        keys = sorted(timing_log.keys())
        for k in keys:
            v = timing_log[k]
            print(f"  {k:40s}: {v:7.1f}s  ({v/60:.1f}min)")
        total = sum(timing_log.values())
        print(f"  {'TOTAL':40s}: {total:7.1f}s  ({total/3600:.2f}hr)")
        pd.DataFrame([{"stage": k, "seconds": v, "minutes": v/60}
                      for k, v in timing_log.items()]).to_csv(
            os.path.join(OUT_DIR, "timing_report.csv"), index=False
        )
        print(f"  Timing saved to timing_report.csv")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # 1. Data Loading (Utilizing the previously unused functions)
    print("\nInitializing Data...")

    ensure_data_exists(MITDB_PATH)
    records = list_records(MITDB_PATH)
    print(f"Found {len(records)} records.")
    
    print("Loading and segmenting beats (this may take a moment)...")

    split_data = load_stratified_temporal_split(records, FS, MITDB_PATH)
    
    X_train_fixed = split_data["X_train"]
    X_val_fixed   = split_data["X_val"]
    X_test_fixed  = split_data["X_test"]
    
    y_train_fixed = split_data["y_train"]
    y_val_fixed   = split_data["y_val"]
    y_test_fixed  = split_data["y_test"]

    print("Normalizing data...")
    global_mu = np.mean(X_train_fixed)
    global_std = np.std(X_train_fixed)

    X_train_fixed = (X_train_fixed - global_mu) / (global_std + 1e-8)
    X_val_fixed   = (X_val_fixed - global_mu) / (global_std + 1e-8)
    X_test_fixed  = (X_test_fixed - global_mu) / (global_std + 1e-8)

    input_len = X_train_fixed.shape[1]
    
    fixed_data = {
        "X_train": X_train_fixed, "y_train": y_train_fixed,
        "X_val":   X_val_fixed,   "y_val":   y_val_fixed,
        "X_test":  X_test_fixed,  "y_test":  y_test_fixed,
        "input_len": input_len
    }

    print("\n" + "="*55)
    print(" DATASET SPLIT STATISTICS (real beats)")
    print("="*55)

    for cls in CLASS_ORDER:
        tr = int(np.sum(y_train_fixed == cls))
        va = int(np.sum(y_val_fixed   == cls))
        te = int(np.sum(y_test_fixed  == cls))

        print(f"  {cls}  |  Train: {tr:>6,}  |  Val: {va:>5,}  |  Test: {te:>5,}")

    print(f"  {'TOT'}  |  Train: {len(y_train_fixed):>6,}  |  Val: {len(y_val_fixed):>5,}  |  Test: {len(y_test_fixed):>5,}")
    
    print("="*55 + "\n")

    # 2. Multi-Seed Loop
    all_seed_results = []
    all_mmd_results = []

    print(f"\nStarting Multi-Seed Study on seeds: {GLOBAL_SEEDS}")
    
    for seed in GLOBAL_SEEDS:
        seed_res, seed_mmd = run_single_seed_experiment(seed, fixed_data)
        all_seed_results.extend(seed_res)
        all_mmd_results.extend(seed_mmd)

        print(f"\n Completed Seed {seed}. Current Aggregate Stats:")
        current_df = pd.DataFrame(all_seed_results)

        summary = current_df.groupby(['Method', 'Ratio'])['Macro_F1'].mean().unstack()
        print(summary)

    # 3. Save Master Results
    df_master = pd.DataFrame(all_seed_results)
    csv_path = os.path.join(OUT_DIR, "final_multi_seed_results.csv")

    df_master.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    print("[Main] Generating MMD quality analysis...")
    df_mmd = pd.DataFrame(all_mmd_results)
    df_mmd.to_csv(os.path.join(OUT_DIR, "mmd_quality_metrics.csv"), index=False)

    mmd_summary = df_mmd.groupby(['Method', 'Class']).agg({
        'MMD': ['mean', 'std'],
        'Diversity': ['mean', 'std'],
        'Fidelity': ['mean', 'std'],
        'CosSim': ['mean', 'std']
    }).reset_index()

    mmd_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) and col[1] 
        else col[0] if isinstance(col, tuple) 
        else col
        for col in mmd_summary.columns.values
    ]

    mmd_summary.rename(columns={
        'MMD_mean': 'MMD Mean', 
        'MMD_std': 'MMD Std',
        'Diversity_mean': 'Diversity Mean', 
        'Diversity_std': 'Diversity Std',
        'Fidelity_mean': 'Fidelity Mean', 
        'Fidelity_std': 'Fidelity Std',
        'CosSim_mean': 'CosSim Mean',
        'CosSim_std': 'CosSim Std'
    }, inplace=True)

    plot_mmd_results(mmd_summary)

    plot_diversity_fidelity(mmd_summary)

    print("MMD analysis complete!")

    # Plot Macro F1, Sensitivity, Accuracy, Improvement
    try:
        plot_comprehensive_results(df_master)
    except Exception as e:
        print(f"Warning: Could not plot comprehensive results. {e}")

    # Plot Variance
    try:
        plot_seed_variance(df_master)
    except Exception as e:
        print(f"Warning: Could not plot seed variance. {e}")

    # 4. Final Scientific Report
    report_scientific_summary(df_master)

    try:
        run_ensemble_evaluation(fixed_data, GLOBAL_SEEDS, mode="DDPM+QLR", ratio=1.0)
    except Exception as e:
        print(f"Error running ensemble: {e}")
    
    print("\n" + "="*60)
    print(" STUDY COMPLETE ")
    print("="*60)