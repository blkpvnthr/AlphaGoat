import os


# (Optional while debugging) run tf.functions eagerly to avoid graph/XLA paths
# tf.config.run_functions_eagerly(True)
# ===============================================================
#  gang-gang-pro.py  — Intraday Conditional WGAN-GP (Pro)
#  • Alpaca 1m/60d + session-aware windowing (no cross-day leakage)
#  • Per-ticker normalization, regime & calendar conditioning
#  • Sector/ticker/regime embeddings
#  • Model: LSTM / TCN / Transformer (switchable)
#  • Stability: GP or R1, grad clip, cosine LR warmup, mixup, feature noise
#  • Hybrid loss: GAN + MMD + autocorr
#  • EMA generator for better samples
#  • Optional Qiskit (cached), optional YAML config
# ===================================
# ===================== USER SETTINGS (overridden by YAML if present) ==========
TICKERS              = ["AAPL", "SPY", "FAKE"]    # FAKE -> FAKE_1, FAKE_2,...

# Data (Alpaca 1m / 60d)
DAYS_BACK            = 90
INTERVAL_MINUTES     = 1
REG_HOURS_ONLY       = False
TIMEZONE             = "America/New_York"

# Indicators
RSI_LEN              = 14
BB_WIN               = 20
BB_NSTD              = 2.0
ATR_WIN              = 14
SRSI_K               = 14
SRSI_D               = 3
SRSI_SLOW            = 3
SRSI_WEIGHT_MID      = 2.0

# Regime features
REALVOL_WIN          = 60            # bars
REALVOL_Q_CUTS       = [0.25,0.5,0.75]
VIX_TICKER           = "^VIX"        # pulled via yfinance
VIX_BUCKETS          = [0.33, 0.66]  # terciles over fetched span
BULL_BEAR_EMA        = 200           # bars (intraday bars) for trend regime

# Conditioning embeddings
USE_TICKER_EMBEDDING = True
TICKER_EMB_DIM       = 8
USE_SECTOR_EMBEDDING = True
SECTOR_EMB_DIM       = 4
USE_REGIME_EMBEDDING = True
REGIME_EMB_DIM       = 4

# Model / training
MODEL_TYPE           = "TCN"         # "LSTM" | "TCN" | "TRANSFORMER"
SEED                 = 0
# --- Model / training ---
AGG_TO_MINUTES = 5
SEQ_LEN       = 70            # 1m bars >= full RTH day
LATENT_DIM    = 32
BATCH_SIZE    = 128
EPOCHS        = 600
N_CRITIC      = 5
USE_R1        = True
GP_LAMBDA     = 10.0           # (unused when USE_R1=True)
R1_GAMMA      = 5.0
LR_BASE       = 5e-5
WARMUP_STEPS  = 1000
CLIPNORM      = 1.0
SAMPLE_EVERY  = 200

# --- Hybrid objectives ---
USE_MMD       = True
MMD_WEIGHT    = 0.25
MMD_SIGMAS    = [0.5, 1.0, 2.0]
USE_ACF_LOSS  = True
ACF_WEIGHT    = 0.10
ACF_MAX_LAG   = 10

# --- Speed ---
ENABLE_XLA_JIT        = False   # <- keep false on Metal
ENABLE_MIXED_PRECISION = False  # stick to float32 on Metal

# Balanced sampling + robustness
BALANCED_SAMPLING    = True
FEATURE_DROPOUT_P    = 0.10
EVENT_FEATURE_NAMES  = [
    "srsi_k_xup_d","srsi_k_xdn_d","srsi_k_xup_80","srsi_k_xdn_20",
    "srsi_k_xup_d_50","srsi_k_xdn_d_50"
]
COND_GAUSS_NOISE_STD = 0.01          # feature noise on conditioning
MIXUP_P              = 0.15          # chance to mix two conditioning rows in a batch
MIXUP_ALPHA          = 0.2

# Hybrid objectives
USE_MMD              = True
MMD_WEIGHT           = 0.5
MMD_SIGMAS           = [0.5, 1.0, 2.0]
USE_ACF_LOSS         = True
ACF_WEIGHT           = 0.2
ACF_MAX_LAG          = 10

# EMA generator
USE_G_EMA            = True
EMA_DECAY            = 0.999

# Quantum (optional — same as before; cached)
USE_QISKIT           = False
N_QUBITS             = 4
Q_FEATURE_SCALE      = 0.5
Q_EMB_DIM            = 2
Q_PCA_WARMUP         = 10000
Q_CACHE_DIR          = "q_cache"

# IBM Runtime (optional)
USE_IBM_BACKEND      = False
IBM_BACKEND_NAME     = "ibmq_qasm_simulator"
IBM_CHANNEL          = "ibm_quantum"
IBM_INSTANCE         = "ibm-q/open/main"
IBM_SHOTS            = 2000
IBM_MAX_BATCH        = 32

# Optional config (YAML) — if present, will override anything above
CONFIG_YAML_PATH     = "configs/pro.yaml"
# ============================================================================

import os, math, warnings, json
os.environ.pop("TF_XLA_FLAGS", None)
os.environ.pop("XLA_FLAGS", None)

import tensorflow as tf
# Force-disable XLA on Metal
try:
    tf.config.optimizer.set_jit(False)
    print("[speed] XLA JIT disabled (Metal).")
except Exception as e:
    print("[speed] Could not change XLA JIT:", e)

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout, BatchNormalization, # type: ignore
                                     LeakyReLU, RepeatVector, TimeDistributed, Concatenate,
                                     Conv1D, LayerNormalization, MultiHeadAttention, Add)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy("float32")
# hard-disable any XLA env flags
os.environ.pop("TF_XLA_FLAGS", None)
os.environ.pop("XLA_FLAGS", None)
# optional: make tf.functions run eagerly while debugging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
ENABLE_XLA_JIT = False  # make sure this is False

try:
    # Force-disable XLA JIT on macOS/Metal
    tf.config.optimizer.set_jit(None)   # or: tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)

    print("[speed] XLA JIT disabled (Metal).")
except Exception as e:
    print("[speed] Could not change XLA JIT:", e)
import yfinance as yf

# Optional YAML override
try:
    pass  # Placeholder for the try block
    import yaml
    if os.path.exists(CONFIG_YAML_PATH):
        with open(CONFIG_YAML_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        globals().update(cfg or {})
        print(f"[config] Loaded YAML overrides from {CONFIG_YAML_PATH}")
except Exception as e:
    print("[config] YAML not loaded:", e)

# ---- Speed-ups ----
if ENABLE_MIXED_PRECISION:
    # Disable mixed precision for debugging stability
    try:
        from tensorflow.keras import mixed_precision # type: ignore
        mixed_precision.set_global_policy('float32')  # override earlier mixed_float16
        print("[speed] Using float32 (no mixed precision) for stability.")
    except Exception:
        pass

    # Optimizers with clipping + lower LR
    GEN_LR = 5e-5
    CRIT_LR = 5e-5
    gen_opt  = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5, beta_2=0.9, clipnorm=1.0)
    crit_opt = tf.keras.optimizers.Adam(learning_rate=CRIT_LR, beta_1=0.5, beta_2=0.9, clipnorm=1.0)

np.random.seed(SEED); tf.random.set_seed(SEED)

# -------------------- Data: Alpaca 1m loader -------------------
# -------------------- Data: Alpaca 1m loader -------------------
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

# --- .env loader (must run before any env var reads) ---
DOTENV_PATH = Path("/Users/abdula1/Library/Mobile Documents/com~apple~CloudDocs/abdula1/Desktop/screener/gang-gang/.env")
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH.as_posix(), override=False)
else:
    print(f"[env] .env not found at {DOTENV_PATH} — relying on existing environment.")

# Preferred feeds (SIP first only if you truly have that subscription)
ALPACA_PRIMARY_FEED  = "sip"
ALPACA_FALLBACK_FEED = "iex"

# -------------- REST helper (single symbol) ---------------
def alpaca_bars_rest(
    symbol: str,
    start_iso: str,
    end_iso: str,
    timeframe: str = "1Min",
    feed: str = "iex",            # default to IEX to avoid SIP 403s
    adjustment: str = "raw",
    limit: int = 10000,
    tz: str = "America/New_York",
    rth_only: bool = True,
    try_fallback: bool = True     # if feed='sip' fails, retry with 'iex'
) -> pd.DataFrame:
    """
    Fetch bars via REST for a single symbol and return a tidy OHLCV DataFrame.
    Requires ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in env or .env.
    """
    api_key = os.getenv("ALPACA_API_KEY_ID")
    api_sec = os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not api_sec:
        raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY")

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_sec,
    }
    params = {
        "timeframe": timeframe,
        "start": start_iso,
        "end":   end_iso,
        "limit": limit,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }

    def _request(_feed: str) -> pd.DataFrame:
        params["feed"] = _feed
        r = requests.get(url, headers=headers, params=params, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            if try_fallback and _feed.lower() == "sip" and r.status_code == 403:
                # typical message: "subscription does not permit querying recent SIP data"
                try:
                    msg = r.json().get("message", "")
                except Exception:
                    msg = ""
                if "subscription" in msg.lower() or "not permit" in msg.lower():
                    print("[alpaca] SIP not permitted; retrying with IEX…")
                    return _request("iex")
            raise

        data = r.json()
        bars = data.get("bars", [])
        df = pd.DataFrame(bars)
        if df.empty:
            return df

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = (
            df.set_index("t")
              .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
              [["Open","High","Low","Close","Volume"]]
              .sort_index()
        )
        df.index = df.index.tz_convert(tz)
        if rth_only:
            df = df.between_time("09:30", "16:00")
        return df

    return _request(feed)

# -------------- SDK helper (recommended) ---------------

def load_alpaca_1m_df(
    ticker: str,
    days: int = 60,
    tz: str = "America/New_York",
    regular_hours_only: bool = True,
    use_prepost: bool = False,
    adjustment: Adjustment = Adjustment.SPLIT,
    feed: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch 1m OHLCV for a single ticker over the last `days` using Alpaca SDK.
    Returns tz-aware OHLCV DataFrame indexed by timestamps.
    Auto-falls back from SIP -> IEX if your subscription forbids SIP.
    """
    key = os.getenv("ALPACA_API_KEY_ID")
    sec = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY env vars.")

    client = StockHistoricalDataClient(key, sec)
    end = pd.Timestamp.now(tz)
    start = end - pd.Timedelta(days=days)

    def _enum_feed(name: str) -> DataFeed:
        return DataFeed.SIP if name.lower() == "sip" else DataFeed.IEX

    feeds_to_try = [feed] if feed else [ALPACA_PRIMARY_FEED, ALPACA_FALLBACK_FEED]
    last_err: Optional[Exception] = None

    for f in feeds_to_try:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,  # 1m
                start=start,
                end=end,
                adjustment=adjustment,
                feed=_enum_feed(f),
            )
            resp = client.get_stock_bars(req)
            df = resp.df
            if df is None or df.empty:
                raise ValueError(f"No Alpaca data for {ticker} with feed={f}")

            # If MultiIndex (symbol, timestamp), slice this ticker cleanly:
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")

            df = df.tz_convert(tz)
            df = (
                df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
                  [["Open","High","Low","Close","Volume"]]
            )
            if regular_hours_only and not use_prepost:
                df = df.between_time("09:30", "16:00")
            df = df[~df.index.duplicated(keep="last")].sort_index()

            if f.lower() != ALPACA_PRIMARY_FEED.lower():
                print(f"[alpaca] Used fallback feed '{f}' for {ticker}.")
            return df

        except APIError as e:
            last_err = e
            msg = str(getattr(e, "error", e)).lower()
            if "subscription" in msg and "sip" in f.lower():
                print(f"[alpaca] SIP not permitted for {ticker}. Falling back to IEX…")
                continue
            raise
        except Exception as e:
            last_err = e
            continue

    # If all attempts failed
    if last_err:
        raise last_err
    raise RuntimeError("Unknown error while fetching Alpaca bars.")
# ------------------------------------------------------------
# ------------------- Indicators & helpers ----------------------
# After you load the first ticker df (post-aggregation), compute a safe SEQ_LEN:
def infer_safe_seq_len(df, floor=32, frac=0.75):
    by_day = df.index.normalize().value_counts()
    if by_day.empty:
        return floor
    L = int(max(floor, min(by_day.min(), int(by_day.median())) * frac))
    return L

# Example usage in your main setup:
first_df = load_alpaca_1m_df(TICKERS[0], days=DAYS_BACK, tz=TIMEZONE, regular_hours_only=REG_HOURS_ONLY)
SEQ_LEN = infer_safe_seq_len(first_df, floor=32, frac=0.75)
print(f"[auto] SEQ_LEN set to {SEQ_LEN} based on available bars.")

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def to_returns(close):
    v = close.values
    return pd.Series(np.diff(np.log(v)), index=close.index[1:])

def rsi_wilders(close, length=14):
    delta = close.diff(); gain = delta.clip(lower=0.0); loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(length).mean(); roll_dn = loss.rolling(length).mean()
    up = pd.Series(index=close.index, dtype=float); dn = pd.Series(index=close.index, dtype=float)
    if len(close) <= length: return pd.Series(np.nan, index=close.index)
    up.iloc[length] = roll_up.iloc[length]; dn.iloc[length] = roll_dn.iloc[length]
    alpha = 1.0/length
    for i in range(length+1, len(close)):
        up.iloc[i] = (1-alpha)*up.iloc[i-1] + alpha*gain.iloc[i]
        dn.iloc[i] = (1-alpha)*dn.iloc[i-1] + alpha*loss.iloc[i]
    rs = up / (dn + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def bollinger_features(close, win=20, nstd=2.0):
    ma = close.rolling(win).mean()
    sd = close.rolling(win).std(ddof=0)
    upper = ma + nstd*sd; lower = ma - nstd*sd
    pb = (close - lower) / (upper - lower + 1e-12)
    bw = (upper - lower) / (ma + 1e-12)
    return pb, bw

def atr_from_df(df, win=14):
    h,l,c = df["High"], df["Low"], df["Close"]; pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(win).mean()

def stoch_rsi_kd(close, rsi_len=14, k_period=14, d_period=3, slowing=3):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_len).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_len).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    rsi_min = rsi.rolling(k_period).min(); rsi_max = rsi.rolling(k_period).max()
    k_raw = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-12)
    wavg = lambda x: np.average(x, weights=np.arange(1, len(x)+1))
    k = k_raw.rolling(slowing).apply(wavg, raw=True)
    d = k.rolling(d_period).apply(wavg, raw=True)
    return k, d

def stoch_rsi_event_features(k, d, weight=2.0):
    diff = k - d; prev = diff.shift(1)
    xup_d = ((prev <= 0) & (diff > 0)).astype(float)
    xdn_d = ((prev >= 0) & (diff < 0)).astype(float)
    prev_k = k.shift(1)
    xup_80 = ((prev_k <= 0.8) & (k > 0.8)).astype(float)
    xdn_20 = ((prev_k >= 0.2) & (k < 0.2)).astype(float)
    xup_d_50 = ((xup_d == 1.0) & (k < 0.5) & (d < 0.5)).astype(float) * weight
    xdn_d_50 = ((xdn_d == 1.0) & (k > 0.5) & (d > 0.5)).astype(float) * weight
    return (xup_d.fillna(0.0), xdn_d.fillna(0.0),
            xup_80.fillna(0.0), xdn_20.fillna(0.0),
            xup_d_50.fillna(0.0), xdn_d_50.fillna(0.0))

def dmi_wilders(df, length=14):
    h,l,c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    ph,pl,pc = h.shift(1), l.shift(1), c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up_move, down_move = h-ph, pl-l
    plus_dm  = ((up_move > 0) & (up_move > down_move)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > 0) & (down_move > up_move)).astype(float) * down_move.clip(lower=0.0)
    def wild_smooth(x,n):
        s=x.rolling(n).sum(); out=pd.Series(index=x.index, dtype=float)
        if len(x)<=n: return pd.Series(np.nan, index=x.index)
        out.iloc[n] = s.iloc[n]; alpha=1.0/n
        for i in range(n+1, len(x)): out.iloc[i] = out.iloc[i-1] - out.iloc[i-1]*alpha + x.iloc[i]
        return out
    tr_w, pdm_w, mdm_w = wild_smooth(tr,length), wild_smooth(plus_dm,length), wild_smooth(minus_dm,length)
    di_plus  = 100.0 * (pdm_w/(tr_w+1e-12))
    di_minus = 100.0 * (mdm_w/(tr_w+1e-12))
    dx  = 100.0 * (abs(di_plus-di_minus)/(di_plus+di_minus+1e-12))
    adx = dx.ewm(alpha=1.0/length, adjust=False, min_periods=length+1).mean()
    return di_plus, di_minus, adx

def zscore_np(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-8), m, s

def standardize_with(x, m, s): return (x - m) / (s + 1e-8)

# Session-aware windowing (no cross-day)
WINDOW_STRIDE = 5
MIN_SESSION_LEN = 32

def make_windows_no_cross(dt_index: pd.DatetimeIndex, arr_2d: np.ndarray, L: int):
    N = len(dt_index)
    if N != len(arr_2d): raise ValueError("Index length mismatch")
    if N <= L: return np.zeros((0, L) + arr_2d.shape[1:], np.float32), []

    try:
        idx_tz = dt_index.tz_convert(TIMEZONE)
    except (TypeError, AttributeError):
        idx_tz = dt_index.tz_localize(TIMEZONE)

    day_keys = np.asarray(idx_tz.date)
    cuts = np.flatnonzero(day_keys[1:] != day_keys[:-1]) + 1
    seg_starts = np.concatenate(([0], cuts))
    seg_ends   = np.concatenate((cuts, [N]))

    windows, idx_pairs = [], []
    stride = max(int(WINDOW_STRIDE), 1)
    for s, e in zip(seg_starts, seg_ends):
        seg_len = e - s
        if seg_len < max(L, MIN_SESSION_LEN):
            continue
        for i in range(s, e - L + 1, stride):
            j = i + L
            windows.append(arr_2d[i:j])
            idx_pairs.append((i, j))

    if not windows:
        return np.zeros((0, L) + arr_2d.shape[1:], np.float32), []
    return np.asarray(windows, np.float32), idx_pairs


def make_windows_sessionwise(idx: pd.DatetimeIndex, arr2d: np.ndarray, L: int):
    # keep your existing call sites unchanged
    return make_windows_no_cross(idx, arr2d, L)

def price_from_returns(seq_norm, anchor_price, ret_mean, ret_std):
    seq = seq_norm.squeeze()*ret_std + ret_mean
    return np.exp(np.cumsum(seq)) * float(anchor_price)

# ----------------- Regimes, calendar, cross-asset --------------
def realized_vol(rets, win=60):
    return rets.rolling(win).std() * np.sqrt(252*390/INTERVAL_MINUTES)

def bull_bear_by_ema(close, span=BULL_BEAR_EMA):
    em = ema(close, span)
    return (close > em).astype(int)  # 1 bull, 0 bear

def weekday_month_features(idx):
    idx = idx.tz_convert(TIMEZONE)
    wd = (idx.weekday / 4.0).values[:,None]  # 0..1
    # sin/cos month (1..12)
    mo = idx.month.values
    sinm = np.sin(2*np.pi*(mo/12.0))[:,None]
    cosm = np.cos(2*np.pi*(mo/12.0))[:,None]
    return wd, sinm, cosm

def load_vix(days=90, tz=TIMEZONE):
    df = yf.download(VIX_TICKER, period=f"{days}d", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty: 
        v = pd.Series(dtype=float)
        cuts = []
    else:
        df = df.sort_index()
        v = df["Close"].tz_localize("UTC").tz_convert(tz)
        cuts = np.quantile(v.dropna(), VIX_BUCKETS).tolist()
    return v, cuts

def align_series(to_idx, s):
    return s.reindex(to_idx).ffill().bfill()

def read_dates_csv(path):
    if os.path.exists(path):
        d = pd.read_csv(path, parse_dates=[0], header=None)[0].dt.tz_localize(TIMEZONE)
        return set(d.dt.date.tolist())
    return set()

# ------------- Expand FAKE tickers + sectors ------------------
fake_counter = 0
tickers = []
for t in TICKERS:
    if str(t).upper() == "FAKE":
        fake_counter += 1
        tickers.append(f"FAKE_{fake_counter}")
    else:
        tickers.append(str(t).upper())
print("Tickers:", tickers)

# crude sector map (extend as needed)
SECTOR_MAP = {
    "AAPL":"TECH","MSFT":"TECH","GOOGL":"TECH","AMZN":"CONS","META":"TECH","NVDA":"TECH",
    "JPM":"FIN","GS":"FIN","XOM":"ENER","CVX":"ENER","TSLA":"AUTO","SPY":"INDEX"
}
SECTORS = sorted(set(SECTOR_MAP.get(t,"UNKNOWN") for t in tickers if not t.startswith("FAKE_")))
if USE_SECTOR_EMBEDDING:
    sector_to_ix = {s:i for i,s in enumerate(sorted(set(list(SECTORS)+["UNKNOWN","INDEX","FAKE"])) )}
    SECTOR_EMB = np.random.normal(0,0.1,size=(len(sector_to_ix), SECTOR_EMB_DIM)).astype(np.float32)

# --------------------- Pull VIX & buckets ---------------------
vix_series, vix_cuts = load_vix(days=180)
if not vix_cuts: vix_cuts = [15,25]  # fallback

# --------------------- Build features per ticker ---------------
ret_vecs, feat_mats, meta, indexes = [], [], [], []

# core features list will grow with regime/calendar channels
core_feat_names = [
    "ema5","ema20","ema50","ema200",
    "bb_pb","bb_bw","atr","rsi",
    "srsi_k_xup_d","srsi_k_xdn_d","srsi_k_xup_80","srsi_k_xdn_20","srsi_k_xup_d_50","srsi_k_xdn_d_50",
    "dmi_di_plus","dmi_di_minus","dmi_adx",
    "bull","rv_q1","rv_q2","rv_q3","rv_q4",
    "vix_level","vix_b1","vix_b2","vix_b3",
    "wd_frac","month_sin","month_cos",
    "fomc_flag","earnings_flag",
    "ones"
]

# optional date flags
FOMC_DATES = read_dates_csv("fomc_dates.csv")  # one date per line
# earnings_{TICKER}.csv for each real ticker if you have it

def load_ticker_df(t):
    if t.startswith("FAKE_"):
        # build synthetic minute bars (rough)
        n = DAYS_BACK * 390
        mu, sig = 0.00005, 0.005
        r = np.random.normal(mu, sig, n)
        idx = pd.date_range(pd.Timestamp.now(TIMEZONE) - pd.Timedelta(days=DAYS_BACK),
                            periods=n, freq=f"{INTERVAL_MINUTES}min", tz=TIMEZONE)
        close = pd.Series(100*np.exp(np.cumsum(r)), index=idx)
        rng = np.maximum(1e-6, pd.Series(np.abs(np.random.normal(0, 0.002, len(close))), index=idx))
        high = close * (1 + rng); low = close * (1 - rng)
        open_ = close.shift(1).fillna(close.iloc[0])
        vol = pd.Series(np.random.randint(1e3, 2e5, len(close)), index=idx)
        df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close, "Adj Close":close, "Volume":vol})
        df = df.between_time("09:30","16:00")
        return df
    else:
        return load_alpaca_1m_df(t, days=DAYS_BACK, tz=TIMEZONE, regular_hours_only=REG_HOURS_ONLY)

for t in tickers:
    df = load_ticker_df(t)
    close = df["Close"].astype(float)

    # base returns & index
    rets = to_returns(close); idx = rets.index
    align = lambda s: s.reindex(idx)

    # indicators
    ema5   = align(ema(close,5)/close - 1.0)
    ema20  = align(ema(close,20)/close - 1.0)
    ema50  = align(ema(close,50)/close - 1.0)
    ema200 = align(ema(close,200)/close - 1.0)

    pb, bw = bollinger_features(close, win=BB_WIN, nstd=BB_NSTD); pb, bw = align(pb), align(bw)
    atr = align(atr_from_df(df, win=ATR_WIN).pct_change())
    rsi = align(rsi_wilders(close, length=RSI_LEN))
    k, d = stoch_rsi_kd(close, rsi_len=RSI_LEN, k_period=SRSI_K, d_period=SRSI_D, slowing=SRSI_SLOW)
    k, d = align(k), align(d)
    (srsi_k_xup_d, srsi_k_xdn_d, srsi_k_xup_80, srsi_k_xdn_20,
     srsi_k_xup_d_50, srsi_k_xdn_d_50) = stoch_rsi_event_features(k, d, weight=SRSI_WEIGHT_MID)
    try:
        di_plus, di_minus, adx = dmi_wilders(df, length=14)
        di_plus, di_minus, adx = align(di_plus), align(di_minus), align(adx)
    except Exception:
        di_plus  = pd.Series(0.0, index=idx); di_minus = pd.Series(0.0, index=idx); adx = pd.Series(0.0, index=idx)

    # Regimes
    bull = align(bull_bear_by_ema(close, span=BULL_BEAR_EMA)).fillna(0.0)
    rv = realized_vol(rets, REALVOL_WIN).reindex(idx)
    q = pd.qcut(rv.rank(method='first'), q=4, labels=[1,2,3,4]).astype(float).fillna(2)
    rv_b = [ (q==i).astype(float) for i in [1,2,3,4] ]

    vix_level = align_series(idx, vix_series).ffill().bfill()
    v1, v2 = vix_cuts
    vix_b1 = (vix_level <= v1).astype(float)
    vix_b2 = ((vix_level>v1)&(vix_level<=v2)).astype(float)
    vix_b3 = (vix_level > v2).astype(float)

    # Calendar
    wd_frac, m_sin, m_cos = weekday_month_features(idx)
    wd_frac = pd.Series(wd_frac.squeeze(), index=idx)
    m_sin   = pd.Series(m_sin.squeeze(), index=idx)
    m_cos   = pd.Series(m_cos.squeeze(), index=idx)

    # FOMC flag, earnings flag
    fomc_flag = pd.Series([1.0 if d.date() in FOMC_DATES else 0.0 for d in idx], index=idx)
    epath = f"earnings_{t}.csv"
    E_DATES = read_dates_csv(epath) if os.path.exists(epath) else set()
    earnings_flag = pd.Series([1.0 if d.date() in E_DATES else 0.0 for d in idx], index=idx)

    ones = pd.Series(1.0, index=idx)

    F_core = pd.concat([
        ema5, ema20, ema50, ema200, pb, bw, atr, rsi,
        srsi_k_xup_d, srsi_k_xdn_d, srsi_k_xup_80, srsi_k_xdn_20, srsi_k_xup_d_50, srsi_k_xdn_d_50,
        di_plus, di_minus, adx,
        bull,
        rv_b[0], rv_b[1], rv_b[2], rv_b[3],
        vix_level, vix_b1, vix_b2, vix_b3,
        wd_frac, m_sin, m_cos,
        fomc_flag, earnings_flag,
        ones
    ], axis=1).fillna(0.0)

    ret_vecs.append(rets.values.astype(float))
    feat_mats.append(F_core.values.astype(float))
    indexes.append(idx)
    meta.append(dict(
        ticker=t,
        is_fake=t.startswith("FAKE_"),
        last_price=float(close.iloc[-1]),
        sector=SECTOR_MAP.get(t, "FAKE" if t.startswith("FAKE_") else "UNKNOWN")
    ))

# ---------------- Per-ticker normalization ---------------------
# returns: per ticker
ret_stats = []
for i, r in enumerate(ret_vecs):
    _, m, s = zscore_np(r)
    ret_stats.append((m, s))

# features: per column, per ticker (optional; here we do per-ticker to follow your ask)
feat_stats = []
for i, F in enumerate(feat_mats):
    col_stats = []
    for j in range(F.shape[1]):
        _, m, s = zscore_np(F[:, j])
        col_stats.append((m, s))
    feat_stats.append(col_stats)

# ---------------- Windows & conditioning -----------------------
uniq_tickers = sorted(set([m["ticker"] for m in meta]))
tic_to_ix = {tk:i for i, tk in enumerate(uniq_tickers)}
n_tickers = len(uniq_tickers)
if USE_TICKER_EMBEDDING:
    TICK_EMB = np.random.normal(0,0.1,size=(n_tickers, TICKER_EMB_DIM)).astype(np.float32)

if USE_REGIME_EMBEDDING:
    # regimes we embed: bull/bear (2), vol quartile (4), vix bucket (3) -> 9 combos max; we keep it simple with 8 ids + fallback
    REGIME_IDS = {}  # (bull, rv_q, vix_b) -> id
    REGIME_EMB = np.random.normal(0,0.1,size=(16, REGIME_EMB_DIM)).astype(np.float32)  # up to 16 combos

core_name_to_idx = {nm: i for i, nm in enumerate(core_feat_names)}
EVENT_COL_IDX = [core_name_to_idx[nm] for nm in EVENT_FEATURE_NAMES if nm in core_name_to_idx]
FZ_DIM_BASE = len(core_feat_names); FZ_DIM = FZ_DIM_BASE

X_by_ticker, C_by_ticker, per_ticker_counts = {}, {}, {}

def make_windows_sessionwise(idx: pd.DatetimeIndex, arr2d: np.ndarray, L: int):
    # Simple wrapper to keep your existing call sites unchanged
    return make_windows_no_cross(idx, arr2d, L)

def regime_id_from_row(rowdict):
    b = int(rowdict["bull"]>0.5)
    # vol quartile one-hot in positions rv_q1..rv_q4
    q = [rowdict.get(f"rv_q{i}",0.0) for i in [1,2,3,4]]
    rvq = int(np.argmax(q))+1
    vb = [rowdict.get("vix_b1",0.0), rowdict.get("vix_b2",0.0), rowdict.get("vix_b3",0.0)]
    vbi = int(np.argmax(vb))+1
    key = (b, rvq, vbi)
    if key not in REGIME_IDS:
        REGIME_IDS[key] = len(REGIME_IDS)%REGIME_EMB.shape[0]
    return REGIME_IDS[key]

for i, m in enumerate(meta):
    rets_i = ret_vecs[i]; F_core = feat_mats[i]; idx = indexes[i]

    # per-ticker standardization
    xz = standardize_with(rets_i, ret_stats[i][0], ret_stats[i][1])
    Fz_cols = [ standardize_with(F_core[:, j], feat_stats[i][j][0], feat_stats[i][j][1]) for j in range(F_core.shape[1]) ]
    Fz = np.stack(Fz_cols, axis=1)

    # Session-aware windows
    Xw, seq_idx = make_windows_sessionwise(idx, xz.reshape(-1,1), SEQ_LEN)
    if len(Xw)==0: continue

    # Optional quantum (not expanded here for brevity) — reuse from earlier versions if needed

    # Fz windows
    Fz_w, _ = make_windows_sessionwise(idx, Fz, SEQ_LEN)

    # Build auxiliary embeddings per window
    # Ticker emb
    if USE_TICKER_EMBEDDING:
        emb_vec = TICK_EMB[tic_to_ix[m["ticker"]]]
        emb_block = np.tile(emb_vec, (len(Xw), SEQ_LEN, 1)).astype(np.float32)
    else:
        oh = np.zeros((len(Xw), SEQ_LEN, n_tickers), dtype=np.float32)
        oh[:,:,tic_to_ix[m["ticker"]]] = 1.0

    # Sector emb
    if USE_SECTOR_EMBEDDING:
        sector_ix = sector_to_ix[m["sector"]]
        sect_vec = SECTOR_EMB[sector_ix]
        sect_block = np.tile(sect_vec, (len(Xw), SEQ_LEN, 1)).astype(np.float32)

    # Regime emb id per timestep (use Fz original values before z-score to extract booleans cleanly)
    if USE_REGIME_EMBEDDING:
        # reconstruct dict per row using original (unscaled) core array for regime flags
        raw = F_core  # unscaled
        kv_names = ["bull","rv_q1","rv_q2","rv_q3","rv_q4","vix_b1","vix_b2","vix_b3"]
        # build ids at row-level, then window them
        ids = []
        for r in range(raw.shape[0]):
            d = dict(zip(core_feat_names, raw[r]))
            ids.append(regime_id_from_row(d))
        ids = np.array(ids)
        # window ids
        ids_w = []
        for s,e in seq_idx:
            ids_w.append(ids[s:e])
        ids_w = np.array(ids_w)  # (N,L)
        reg_block = REGIME_EMB[ids_w]  # (N,L,REGIME_EMB_DIM)

    # is_fake block
    is_fake_w = np.full((len(Xw), SEQ_LEN, 1), 1.0 if m["is_fake"] else 0.0, dtype=np.float32)

    # Compose conditioning
    parts = [Fz_w]
    if USE_TICKER_EMBEDDING: parts.append(emb_block)
    else: parts.append(oh)
    if USE_SECTOR_EMBEDDING: parts.append(sect_block)
    if USE_REGIME_EMBEDDING: parts.append(reg_block)
    parts.append(is_fake_w)
    Cw = np.concatenate(parts, axis=-1)

    X_by_ticker[m["ticker"]] = Xw
    C_by_ticker[m["ticker"]] = Cw
    per_ticker_counts[m["ticker"]] = len(Xw)

if not X_by_ticker:
    raise RuntimeError("No training windows constructed.")

cond_dim = next(iter(C_by_ticker.values())).shape[-1]
FZ_DIM = feat_mats[0].shape[1]  # core feat dim (z-scored) — used for event dropout boundary

print(f"Cond dim: {cond_dim} | Core feat dim: {FZ_DIM}")
print("Per-ticker windows:", per_ticker_counts)

# -------------------- Batching (balanced) ----------------------
def get_balanced_batches(X_by, C_by, bs):
    tic = list(X_by.keys()); Tn = len(tic)
    base = bs // Tn; rem = bs - base*Tn
    shares = [base + (1 if k<rem else 0) for k in range(Tn)]
    total = sum(len(X_by[t]) for t in tic)
    nb = max(1, total//bs)
    for _ in range(nb):
        xs, cs = [], []
        for k,t in enumerate(tic):
            n = shares[k]; N = len(X_by[t])
            idx = np.random.randint(0, N, size=n)
            xs.append(X_by[t][idx]); cs.append(C_by[t][idx])
        bx = np.concatenate(xs,0); bc = np.concatenate(cs,0)
        perm = np.random.permutation(len(bx))
        yield bx[perm], bc[perm]

# ----------------- Robustness (dropout, noise, mixup) ----------
def apply_event_dropout(bc, event_cols, p, fz_dim):
    if p<=0.0 or not event_cols: return bc
    B,L,F = bc.shape
    keep = (np.random.rand(B, len(event_cols)) > p).astype(np.float32)
    mask = np.ones((B,1,F), dtype=np.float32)
    for j,col in enumerate(event_cols):
        if col<fz_dim: mask[:,0,col] = keep[:,j]
    return bc*mask

def add_cond_noise(bc, std):
    if std<=0: return bc
    return bc + np.random.normal(0, std, size=bc.shape).astype(np.float32)

def maybe_mixup(bx, bc, p: float, alpha: float):
    """
    bx: (B, L, 1)  Tensor
    bc: (B, L, F)  Tensor
    With prob p, apply mixup across the batch using Beta(alpha, alpha) weight.
    """
    bx = tf.convert_to_tensor(bx, dtype=tf.float32)
    bc = tf.convert_to_tensor(bc, dtype=tf.float32)

    if p <= 0.0 or alpha <= 0.0:
        return bx, bc

    u = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)

    def _do_mixup():
        # Sample lam ~ Beta(alpha, alpha) via Gamma(alpha,1)
        g1 = tf.random.gamma(shape=[1], alpha=alpha, dtype=tf.float32)
        g2 = tf.random.gamma(shape=[1], alpha=alpha, dtype=tf.float32)
        lam = g1 / (g1 + g2)  # shape (1,)
        # broadcast to (B,1,1) for (B,L,1) and (B,L,F)
        B = tf.shape(bx)[0]
        lam_b = tf.reshape(lam, [1, 1, 1])
        lam_b = tf.tile(lam_b, [B, 1, 1])

        idx = tf.random.shuffle(tf.range(B))                  # tensor indices
        bx_shuf = tf.gather(bx, idx, axis=0)
        bc_shuf = tf.gather(bc, idx, axis=0)

        bx_mix = lam_b * bx + (1.0 - lam_b) * bx_shuf
        # for bc (B,L,F), lam broadcast over last dim automatically
        bc_mix = lam_b * bc + (1.0 - lam_b) * bc_shuf
        return bx_mix, bc_mix

    return tf.cond(u < p, _do_mixup, lambda: (bx, bc))


# --------------------- Models (LSTM/TCN/Transformer) -----------
def block_tcn(x, ch=128, k=3, d=1, dropout=0.05):
    h = Conv1D(ch, k, padding="causal", dilation_rate=d)(x)
    h = LayerNormalization()(h); h = tf.keras.activations.gelu(h)
    h = Dropout(dropout)(h)
    h = Conv1D(ch, k, padding="causal", dilation_rate=d)(h)
    h = LayerNormalization()(h); h = tf.keras.activations.gelu(h)
    if x.shape[-1] != ch:
        x = Conv1D(ch, 1, padding="same")(x)
    return Add()([x, 0.5*h])   # residual scaling


D_MODEL = 128   # transformer channel width

def block_transformer(x, heads=4, dff=256, dropout=0.1):
    # project to D_MODEL first (shape-safe)
    h = Dense(D_MODEL)(x)
    # causal self-attention (no future leakage)
    attn = MultiHeadAttention(num_heads=heads, key_dim=D_MODEL // heads, dropout=dropout)
    a = attn(h, h, use_causal_mask=True)
    h = LayerNormalization()(h + a)                    # Pre-LN style residual
    ff = Dense(dff, activation="gelu")(h)
    ff = Dropout(dropout)(ff)
    ff = Dense(D_MODEL)(ff)
    h = LayerNormalization()(h + ff)                   # residual
    return h


def build_generator(Fdim):
    z_in = Input(shape=(LATENT_DIM,))
    c_in = Input(shape=(SEQ_LEN, Fdim))
    # context squeeze
    ctx = tf.keras.layers.GlobalAveragePooling1D()(c_in)
    ctx = Dense(64, activation="relu")(ctx)
    h = Concatenate()([z_in, ctx])
    h = Dense(128, activation="relu")(h)
    h = RepeatVector(SEQ_LEN)(h)
    h = Concatenate(axis=-1)([h, c_in])
    if MODEL_TYPE=="LSTM":
        h = LSTM(128, return_sequences=True)(h)
        h = LSTM(64, return_sequences=True)(h)
    elif MODEL_TYPE=="TCN":
        h = block_tcn(h, ch=128, k=3, d=1)
        h = block_tcn(h, ch=128, k=3, d=2)
        h = block_tcn(h, ch=64,  k=3, d=4)
    else:  # TRANSFORMER
        h = Dense(128, activation="relu")(h)
        h = block_transformer(h, heads=4, dff=128)
        h = block_transformer(h, heads=4, dff=128)
    h = TimeDistributed(Dense(32, activation="relu"))(h)
    out = TimeDistributed(Dense(1, activation='tanh'))(h)
    return Model([z_in, c_in], out, name="G")

# (Optional) tiny spectral norm for Conv/Dense
class SpectralDense(Dense):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight("u", shape=(1, self.kernel.shape[-1]),
                                 initializer="random_normal", trainable=False)
    def call(self, inputs):
        # power iteration
        w = self.kernel
        u = self.u
        v = tf.linalg.l2_normalize(tf.matmul(u, tf.transpose(w)))
        u = tf.linalg.l2_normalize(tf.matmul(v, w))
        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
        w_sn = w / sigma
        self.kernel.assign(w_sn)
        self.u.assign(u)
        return super().call(inputs)

def build_critic(Fdim):
    x_in = Input(shape=(SEQ_LEN, 1))
    c_in = Input(shape=(SEQ_LEN, Fdim))
    h = Concatenate(axis=-1)([x_in, c_in])

    if MODEL_TYPE == "LSTM":
        h = LSTM(256, return_sequences=True)(h); h = Dropout(0.2)(h)
        h = LSTM(128)(h); h = Dropout(0.2)(h)
    elif MODEL_TYPE == "TCN":
        for d in [1, 2, 4, 8]:
            h = block_tcn(h, ch=D_MODEL, k=3, d=d)
        h = tf.keras.layers.GlobalAveragePooling1D()(h)
    else:  # TRANSFORMER critic
        h = Dense(D_MODEL)(h)
        h = block_transformer(h, heads=4, dff=4*D_MODEL)
        h = tf.keras.layers.GlobalAveragePooling1D()(h)

    h = LayerNormalization()(h)
    h = Dense(128)(h); h = LeakyReLU(0.2)(h)
    h = Dropout(0.1)(h)
    out = Dense(1)(h)  # linear
    return Model([x_in, c_in], out, name="C")

G = build_generator(cond_dim)
C = build_critic(cond_dim)

# EMA generator
if USE_G_EMA:
    G_EMA = tf.keras.models.clone_model(G); G_EMA.set_weights(G.get_weights())
def ema_update():
    if not USE_G_EMA: return
    w, we = G.get_weights(), G_EMA.get_weights()
    G_EMA.set_weights([EMA_DECAY*e + (1-EMA_DECAY)*v for e, v in zip(we, w)])

# Schedules & optimizers
total_steps = EPOCHS * max(1, sum(per_ticker_counts.values())//BATCH_SIZE)
lr_sched = CosineDecay(initial_learning_rate=LR_BASE, decay_steps=max(1,total_steps-WARMUP_STEPS), alpha=0.1)
def lr_with_warmup(step):
    if step < WARMUP_STEPS:
        return LR_BASE * (step / max(1.0, WARMUP_STEPS))
    return lr_sched(step - WARMUP_STEPS).numpy()

opt_c = Adam(learning_rate=LR_BASE, beta_1=0.0, beta_2=0.9, clipnorm=CLIPNORM)
opt_g = Adam(learning_rate=LR_BASE, beta_1=0.0, beta_2=0.9, clipnorm=CLIPNORM)


# ---------------------- Loss helpers ---------------------------
def mmd_rbf(x, y, sigmas):
    # x,y: (B,L,1) returns
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    y = tf.reshape(y, [tf.shape(y)[0], -1])
    def pdist(a, b):
        a2 = tf.reduce_sum(a*a,1,keepdims=True)
        b2 = tf.reduce_sum(b*b,1,keepdims=True)
        return a2 - 2*tf.matmul(a,b,transpose_b=True) + tf.transpose(b2)
    Kxx = 0; Kyy = 0; Kxy = 0
    for s in sigmas:
        gamma = 1.0/(2.0*(s**2))
        Kxx += tf.exp(-gamma * pdist(x,x))
        Kyy += tf.exp(-gamma * pdist(y,y))
        Kxy += tf.exp(-gamma * pdist(x,y))
    m = tf.cast(tf.shape(x)[0], tf.float32); n = tf.cast(tf.shape(y)[0], tf.float32)
    mmd = (tf.reduce_sum(Kxx) - tf.reduce_sum(tf.linalg.diag_part(Kxx)))/(m*(m-1)+1e-8) \
        + (tf.reduce_sum(Kyy) - tf.reduce_sum(tf.linalg.diag_part(Kyy)))/(n*(n-1)+1e-8) \
        - 2*tf.reduce_sum(Kxy)/(m*n+1e-8)
    return mmd

def acf_penalty(x, y, max_lag=10):
    """
    x,y: Tensor (B, L, 1). Returns scalar penalty matching ACFs up to max_lag.
    Robust to zero-variance windows and short sequences.
    """
    EPS = tf.constant(1e-8, tf.float32)

    # Ensure float32 tensors with last dim squeezed
    x = tf.cast(tf.squeeze(x, -1), tf.float32)  # (B, L)
    y = tf.cast(tf.squeeze(y, -1), tf.float32)  # (B, L)

    B = tf.shape(x)[0]
    L = tf.shape(x)[1]

    # Effective max lag cannot exceed L-1
    k_eff = tf.maximum(1, tf.minimum(max_lag, tf.math.maximum(1, L - 1)))

    def norm_acf(z, k):
        # center
        z = z - tf.reduce_mean(z, axis=1, keepdims=True)  # (B, L)
        var = tf.reduce_mean(z * z, axis=1, keepdims=True)
        var = tf.where(var < EPS, EPS, var)               # avoid div-by-zero

        acfs = []
        # NOTE: for very short L, some lags may be invalid; guard loop bounds
        for lag in range(1, int(z.shape[1]) if isinstance(z.shape[1], int) else 1024):
            # stop when lag > k or lag >= L
            if_lag_ok = tf.less_equal(lag, k)
            if_not_past_len = tf.less(lag, tf.shape(z)[1])
            if not (isinstance(k_eff, tf.Tensor) or isinstance(L, tf.Tensor)):
                pass
            if isinstance(k, tf.Tensor):
                # dynamic stop via tf.cond is expensive; just break when we collected enough
                pass
            a = z[:, :-lag]
            b = z[:, lag:]
            # If shapes are dynamic, ensure at least one column remains
            ac = tf.reduce_mean(a * b, axis=1, keepdims=True) / var  # (B,1)
            acfs.append(ac)
            if len(acfs) >= int(k.numpy()) if isinstance(k, tf.Tensor) and k.shape == () else len(acfs) >= k:
                break

        # Concatenate; if no lags collected (extreme short L), return zeros
        if not acfs:
            return tf.zeros((B, 1), dtype=tf.float32)
        out = tf.concat(acfs, axis=1)
        # Clean any residual non-finites
        out = tf.where(tf.math.is_finite(out), out, tf.zeros_like(out))
        return out

    # Compute with dynamic k_eff
    k_eff_val = tf.cast(k_eff, tf.int32)
    ax = norm_acf(x, k_eff_val)  # (B, k_eff)
    ay = norm_acf(y, k_eff_val)

    diff = ax - ay
    diff = tf.where(tf.math.is_finite(diff), diff, tf.zeros_like(diff))
    pen = tf.reduce_mean(diff * diff)

    # Assert finite to fail fast in debug
    tf.debugging.assert_all_finite(pen, "acf_penalty produced NaN/Inf")
    return pen

# ================== GAN steps (define BEFORE training loop) ==================
@tf.function(experimental_compile=False) # add this on ALL of these
def gradient_penalty(real_x, fake_x, cond):
    bs  = tf.shape(real_x)[0]
    eps = tf.random.uniform([bs, 1, 1], 0., 1., dtype=real_x.dtype)
    inter = eps * real_x + (1. - eps) * fake_x
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = C([inter, cond], training=True)
    grads = t.gradient(pred, inter)
    grads = tf.reshape(grads, [bs, -1])
    gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
    tf.debugging.assert_all_finite(gp, "GP is NaN/Inf")
    return gp

@tf.function(experimental_compile=False) 
def r1_penalty(real_x, cond):
    with tf.GradientTape() as t:
        t.watch(real_x)
        pred = C([real_x, cond], training=True)
    grads = t.gradient(pred, real_x)
    r1 = tf.reduce_mean(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    tf.debugging.assert_all_finite(r1, "R1 is NaN/Inf")
    return r1

@tf.function(experimental_compile=False) 
def critic_step(real_x, cond):
    real_x = tf.cast(real_x, tf.float32)
    cond   = tf.cast(cond,   tf.float32)
    bs = tf.shape(real_x)[0]
    z  = tf.random.normal((bs, LATENT_DIM), dtype=tf.float32)

    with tf.GradientTape() as tape:
        fake_x = G([z, cond], training=True)
        c_real = C([real_x, cond], training=True)
        c_fake = C([fake_x, cond], training=True)
        wass = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real)
        if USE_R1:
            reg = (R1_GAMMA / 2.0) * r1_penalty(real_x, cond)
            loss = wass + reg
            gp = reg
        else:
            gp = GP_LAMBDA * gradient_penalty(real_x, fake_x, cond)
            loss = wass + gp
        tf.debugging.assert_all_finite(loss, "Critic loss NaN/Inf")
    grads = tape.gradient(loss, C.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    opt_c.apply_gradients(zip(grads, C.trainable_variables))
    return loss, wass, gp

@tf.function(experimental_compile=False) 
def generator_step(real_x, cond):
    real_x = tf.cast(real_x, tf.float32)
    cond   = tf.cast(cond,   tf.float32)
    bs = tf.shape(cond)[0]
    z  = tf.random.normal((bs, LATENT_DIM), dtype=tf.float32)
    with tf.GradientTape() as tape:
        fake_x = G([z, cond], training=True)
        c_fake = C([fake_x, cond], training=True)
        adv = -tf.reduce_mean(c_fake)

        add = tf.constant(0.0, dtype=tf.float32)
        if USE_MMD:
            add += tf.cast(MMD_WEIGHT, tf.float32) * mmd_rbf(fake_x, real_x, tf.constant(MMD_SIGMAS, dtype=tf.float32))
        if USE_ACF_LOSS:
            add += tf.cast(ACF_WEIGHT, tf.float32) * acf_penalty(fake_x, real_x, max_lag=ACF_MAX_LAG)

        loss = adv + add
        tf.debugging.assert_all_finite(loss, "Generator loss NaN/Inf")
    grads = tape.gradient(loss, G.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    opt_g.apply_gradients(zip(grads, G.trainable_variables))
    return loss, adv, add
# ----------------- Helper: finite check -----------------
def _finite(x):
    try:
        if tf.is_tensor(x):
            x = tf.reduce_mean(tf.cast(x, tf.float32))
        return np.isfinite(float(x))
    except Exception:
        return False

# ----------------- mini-batch generator -----------------
def training_batches():
    """Yield one epoch of mini-batches (balanced or pooled)."""
    if BALANCED_SAMPLING:
        yield from get_balanced_batches(X_by_ticker, C_by_ticker, BATCH_SIZE)
        return
    X_all = np.concatenate(list(X_by_ticker.values()), axis=0)
    C_all = np.concatenate(list(C_by_ticker.values()), axis=0)
    nb = max(1, len(X_all) // BATCH_SIZE)
    for _ in range(nb):
        idx = np.random.randint(0, len(X_all), size=BATCH_SIZE)
        yield X_all[idx], C_all[idx]

# -------------------------- Training ---------------------------
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
def _last(xs, default=float('10')):
    """Return the last element of a list or a default if empty."""
    return xs[-1] if xs else default
total_samples = sum(len(x) for x in X_by_ticker.values())
steps_per_epoch = max(1, total_samples // BATCH_SIZE)
print(f"[train] steps_per_epoch={steps_per_epoch}, batch_size={BATCH_SIZE}")
assert steps_per_epoch > 0, "No training steps! Check batch size vs. dataset size."

print("Training…")
hist = {"c": [], "g": [], "w": [], "gp": [], "adv": [], "aux": []}
step = 0

for epoch in range(1, EPOCHS + 1):
    for bx, bc in training_batches():
        # numpy -> tensors (consistent dtype)
        bx = tf.convert_to_tensor(bx, dtype=tf.float32)
        bc = tf.convert_to_tensor(bc, dtype=tf.float32)

        # conditioning robustness
        bc = apply_event_dropout(bc, EVENT_COL_IDX, FEATURE_DROPOUT_P, FZ_DIM)
        bc = add_cond_noise(bc, COND_GAUSS_NOISE_STD)
        bx, bc = maybe_mixup(bx, bc, MIXUP_P, MIXUP_ALPHA)

        # Critic updates
        for _ in range(N_CRITIC):
            lr = lr_with_warmup(step)
            try:
                opt_c.lr.assign(lr); opt_g.lr.assign(lr)
            except Exception:
                opt_c.learning_rate = lr; opt_g.learning_rate = lr

            c_loss, w, gp = critic_step(bx, bc)
            if _finite(c_loss) and _finite(w) and _finite(gp):
                hist["c"].append(float(c_loss)); hist["w"].append(float(w)); hist["gp"].append(float(gp))
            step += 1

            # quick heartbeat every 10 steps
            if step % 10 == 0:
                print(f"[step {step}] c={_last(hist['c']):.4f} g={_last(hist['g']):.4f} w={_last(hist['w']):.4f} gp={_last(hist['gp']):.4f}")
                anchor = float(np.median([m["last_price"] for m in meta]))
                z = tf.random.normal((BATCH_SIZE, LATENT_DIM), dtype=tf.float32)
                G_use = G_EMA if USE_G_EMA else G
                fake = tf.squeeze(G_use([z, bc], training=False), axis=-1).numpy()

                plt.figure(figsize=(9, 5))
                for r in tf.squeeze(bx, axis=-1).numpy()[:5]:
                    plt.plot(price_from_returns(r, anchor, ret_stats[0][0], ret_stats[0][1]), alpha=0.7)
                for f in fake[:5]:
                    plt.plot(price_from_returns(f, anchor, ret_stats[0][0], ret_stats[0][1]), linestyle='--', alpha=0.7)
                plt.title(f'Pro Cond WGAN — step {step}')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f'samples_{step:06d}.png'), dpi=140)
                plt.close()

        # Generator update
        lr = lr_with_warmup(step)
        try:
            opt_c.lr.assign(lr); opt_g.lr.assign(lr)
        except Exception:
            opt_c.learning_rate = lr; opt_g.learning_rate = lr

        g_loss, adv, aux = generator_step(bx, bc)
        if _finite(g_loss): hist["g"].append(float(g_loss))
        if _finite(adv):    hist["adv"].append(float(adv))
        if _finite(aux):    hist["aux"].append(float(aux))
        ema_update()

    LOG_EVERY_EPOCH = 2 # faster feedback
    if epoch % LOG_EVERY_EPOCH == 0:
        def _last(lst): return lst[-1] if lst else float("nan")
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"C {_last(hist['c']):.4f} | G {_last(hist['g']):.4f} | "
              f"W {_last(hist['w']):.4f} | REG {_last(hist['gp']):.4f} | "
              f"ADV {_last(hist['adv']):.4f} | AUX {_last(hist['aux']):.4f} | LR {lr:.2e}")
        print(_last(hist['w']), _last(hist['gp']))

# ----------------- diagnostics -----------------
if hist["w"]:
    plt.figure(figsize=(8,5)); plt.plot(hist["w"], label="W")
    if hist["gp"]: plt.plot(hist["gp"], label="Reg (GP/R1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "critic_stats.png"), dpi=140); plt.close()

if hist["c"] or hist["g"] or hist["adv"] or hist["aux"]:
    plt.figure(figsize=(8,5))
    if hist["c"]:   plt.plot(hist["c"], label="Critic")
    if hist["g"]:   plt.plot(hist["g"], label="Generator")
    if hist["adv"]: plt.plot(hist["adv"], label="G adv")
    if hist["aux"]: plt.plot(hist["aux"], label="G aux")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "losses.png"), dpi=140); plt.close()

# ----------------------- Save + diagnostics --------------------
(G_EMA if USE_G_EMA else G).save(os.path.join(OUT_DIR, "G_best.keras"))
G.save(os.path.join(OUT_DIR, "G_latest.keras")); C.save(os.path.join(OUT_DIR, "C_latest.keras"))

plt.figure(figsize=(8,5)); plt.plot(hist["w"], label="W"); plt.plot(hist["gp"], label="Reg (GP/R1)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "critic_stats.png"), dpi=140); plt.close()

plt.figure(figsize=(8,5)); plt.plot(hist["c"], label="Critic"); plt.plot(hist["g"], label="Generator")
plt.plot(hist["adv"], label="G adv"); plt.plot(hist["aux"], label="G aux"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "losses.png"), dpi=140); plt.close()

# --------------- Evaluation (quick sanity metrics) -------------
def ecdf(x):
    xs = np.sort(x); ys = np.arange(1,len(x)+1)/len(x); return xs, ys

def ks_distance(x,y):
    xs, Fy = ecdf(y); xs2, Fx = ecdf(x)
    # align grid
    grid = np.sort(np.concatenate([xs,xs2]))
    Fy_i = np.interp(grid, xs, Fy); Fx_i = np.interp(grid, xs2, Fx)
    return np.max(np.abs(Fx_i - Fy_i))

def acf_vec(x, lag=10):
    x = x - x.mean(); v = (x**2).mean()+1e-12
    return np.array([np.dot(x[:-k], x[k:])/(len(x)-k)/v for k in range(1,lag+1)])

def pnl_ma_crossover(prices, fast=10, slow=30):
    import pandas as pd
    p = pd.Series(prices)
    f = p.rolling(fast).mean(); s = p.rolling(slow).mean()
    pos = (f>s).astype(int).shift(1).fillna(0)
    ret = p.pct_change().fillna(0)
    pnl = (pos*ret+1).cumprod()
    return pnl.iloc[-1]-1, pnl

# make a real batch and a fake batch
some_ticker = next(iter(C_by_ticker.keys()))
cond_seed = C_by_ticker[some_ticker][:min(256, len(C_by_ticker[some_ticker]))]
z = tf.random.normal((len(cond_seed), LATENT_DIM))
G_use = G_EMA if USE_G_EMA else G
# ensure both inputs are tensors with the same dtype
z_tf    = tf.convert_to_tensor(z, dtype=tf.float32)
cond_tf = tf.convert_to_tensor(cond_seed, dtype=tf.float32)

gen_norm = G_use([z_tf, cond_tf], training=False)
gen_norm = tf.squeeze(gen_norm, axis=-1).numpy()

real_norm = X_by_ticker[some_ticker][:len(cond_seed)].squeeze(axis=-1)

# KS on pooled returns
ks = ks_distance(real_norm.flatten(), gen_norm.flatten())

# ACF distance
acf_r = acf_vec(real_norm.flatten(), ACF_MAX_LAG)
acf_g = acf_vec(gen_norm.flatten(), ACF_MAX_LAG)
acf_dist = np.sqrt(((acf_r - acf_g)**2).mean())

# PnL sanity
anchor = 100.0
real_p = [price_from_returns(r, anchor, ret_stats[0][0], ret_stats[0][1]) for r in real_norm[:8]]
gen_p  = [price_from_returns(g, anchor, ret_stats[0][0], ret_stats[0][1]) for g in gen_norm[:8]]
pnl_r  = [pnl_ma_crossover(p)[0] for p in real_p]
pnl_g  = [pnl_ma_crossover(p)[0] for p in gen_p]

with open(os.path.join(OUT_DIR, "eval_summary.json"), "w") as f:
    json.dump({"ks_distance": float(ks),
               "acf_distance": float(acf_dist),
               "real_pnl_mean": float(np.mean(pnl_r)),
               "gen_pnl_mean": float(np.mean(pnl_g))}, f, indent=2)
print("Eval:", {"ks":ks, "acf_dist":acf_dist, "real_pnl_mean":np.mean(pnl_r), "gen_pnl_mean":np.mean(pnl_g)})

pd.DataFrame(gen_norm).to_csv(os.path.join(OUT_DIR, "generated_returns.csv"), index=False)
