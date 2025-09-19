# rl_trader.py
# All-config-via-variables + trade plot with buy/sell markers + inverse return scaling.

import os, math, random, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

# Alpaca Market Data v2
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

# yfinance only to pull daily VIX if your sidecar/core names include VIX items
import yfinance as yf

# ========================= USER CONFIG (variables) =========================
GAN_OUT_DIR         = "output"   # folder with G_best.keras/G_latest.keras (+ conditioning_meta.json)
USE_GAN_SAMPLER     = True     # True: train on GAN-sampled returns using Alpaca conditioning; False: train only on real Alpaca windows
GAN_MIX             = 1.0      # fraction of training episodes from GAN (0..1). If <1, remaining are real Alpaca windows

ALPACA_TICKERS      = ["AAPL", "SPY"]  # tickers for conditioning windows and (optionally) real training windows
ALPACA_DAYS         = 30               # days back for 1m bars
ALPACA_TZ           = "America/New_York"
ALPACA_RTH_ONLY     = True

# DQN training
EPISODES            = 400
MAX_STEPS           = 20000     # upper bound on total env steps
BATCH_SIZE          = 128
GAMMA               = 0.99
LR                  = 1e-3
REPLAY_SIZE         = 50_000
TARGET_UPDATE_EVERY = 1000
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 10_000
SEED                = 0

# Trading costs / shaping (basis points => 1bp = 0.01%)
TCOST_BPS           = 1.0       # cost when position changes
POS_PENALTY_BPS     = 0.0       # holding penalty per |pos|

# Evaluation on real Alpaca data
EVAL_TICKER         = "SPY"
EVAL_DAYS           = 30

# Output
OUT_DIR             = None      # default: <GAN_OUT_DIR>/rl_agent
SAVE_TRADES_CSV     = True
SAVE_TRADE_PLOT     = True

# ==========================================================================

# -------------------- Utilities & indicators --------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def to_returns(close):
    v = close.values
    return pd.Series(np.diff(np.log(v)), index=close.index[1:])

def price_from_returns(norm_ret_seq, anchor_price, ret_mean, ret_std):
    """Rebuild price path from normalized returns + anchor (first price)."""
    seq = norm_ret_seq * (ret_std + 1e-12) + ret_mean
    return np.exp(np.cumsum(seq)) * float(anchor_price)

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

def time_of_day_feature(idx, tz="America/New_York"):
    idx = idx.tz_convert(tz); mins = idx.hour*60 + idx.minute
    start = 9*60 + 30
    return pd.Series(np.clip(mins - start, 0, 390) / 390.0, index=idx, name="tod")

def zscore_np(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-8), m, s

# Regime helpers (minimal; only if your sidecar expects them)
def realized_vol_series(returns, win=60, minutes=1):
    return returns.rolling(win).std() * np.sqrt(252*390/minutes)

def load_vix_series(days=180, tz="America/New_York"):
    try:
        vix = yf.download("^VIX", period=f"{days}d", interval="1d", progress=False, auto_adjust=False)["Close"]
        return vix.tz_localize("UTC").tz_convert(tz)
    except Exception:
        return pd.Series(dtype=float)

def regime_ids_from_flags(idx, close, returns, vix_series, vix_cuts=(15,25), rv_win=60):
    bb = (close > close.ewm(span=200, adjust=False).mean()).astype(int).reindex(idx).fillna(0).values
    rv = realized_vol_series(pd.Series(returns, index=idx), win=rv_win).reindex(idx).fillna(method="ffill")
    try:
        q = pd.qcut(rv.rank(method='first'), q=4, labels=[1,2,3,4]).astype(int)
    except Exception:
        q = pd.Series(2, index=idx)
    vix_aligned = vix_series.reindex(idx.date, method="ffill") if not vix_series.empty else pd.Series(20, index=pd.Index(idx.date))
    v1, v2 = vix_cuts if isinstance(vix_cuts, (list,tuple)) else (15,25)
    vb = np.where(vix_aligned.values <= v1, 1, np.where(vix_aligned.values <= v2, 2, 3))
    return (bb*12 + (q.values-1)*3 + (vb-1)).astype(int)  # 0..23

# -------------------- Alpaca loader --------------------
def alpaca_df(ticker, days=30, tz="America/New_York", rth_only=True, use_prepost=False,
              adjustment=Adjustment.SPLIT, feed=DataFeed.SIP):
    key = os.getenv("ALPACA_API_KEY_ID"); sec = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY environment variables.")
    client = StockHistoricalDataClient(key, sec)
    end = pd.Timestamp.now(tz)
    start = end - pd.Timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
                           start=start, end=end, adjustment=adjustment, feed=feed)
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    df = df.tz_convert(tz).rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})[
        ["Open","High","Low","Close","Volume"]
    ]
    if rth_only and not use_prepost: df = df.between_time("09:30","16:00")
    return df[~df.index.duplicated(keep="last")].sort_index()

# -------------------- Sidecar & features --------------------
def load_sidecar_meta(gan_out_dir):
    path = os.path.join(gan_out_dir, "conditioning_meta.json")
    if not os.path.exists(path):
        print("[sidecar] No conditioning_meta.json found; using computed stats and zero-padded embeddings.")
        return None
    with open(path, "r") as f:
        meta = json.load(f)
    print("[sidecar] Loaded", path)
    return meta

def reorder_or_pad_features(F_core_np, have_names, want_names):
    idx_map = {n:i for i,n in enumerate(have_names)}
    cols = []
    for w in want_names:
        if w in idx_map:
            cols.append(F_core_np[:, idx_map[w]][:, None])
        else:
            cols.append(np.zeros((F_core_np.shape[0], 1), dtype=F_core_np.dtype))
    return np.concatenate(cols, axis=1)

def stats_apply_per_ticker(F_np, feat_stats_for_ticker):
    out = np.empty_like(F_np, dtype=np.float32)
    for j in range(F_np.shape[1]):
        if feat_stats_for_ticker and j < len(feat_stats_for_ticker):
            m, s = feat_stats_for_ticker[j]
        else:
            m, s = float(np.nanmean(F_np[:,j])), float(np.nanstd(F_np[:,j]) + 1e-8)
        out[:,j] = (F_np[:,j] - m) / (s + 1e-8)
    return out

def build_features_from_df(df, tz="America/New_York",
                           rsi_len=14, bb_win=20, bb_nstd=2.0, atr_win=14,
                           srsi_k=14, srsi_d=3, srsi_slow=3, srsi_w=2.0,
                           include_regime=False, vix_series=None):
    close = df["Close"].astype(float)
    rets = to_returns(close); idx = rets.index
    align = lambda s: s.reindex(idx)

    ema5   = align(ema(close,5)/close - 1.0)
    ema20  = align(ema(close,20)/close - 1.0)
    ema50  = align(ema(close,50)/close - 1.0)
    ema200 = align(ema(close,200)/close - 1.0)
    pb, bw = bollinger_features(close, win=bb_win, nstd=bb_nstd); pb, bw = align(pb), align(bw)
    atr = align(atr_from_df(df, win=atr_win).pct_change())
    rsi = align(rsi_wilders(close, length=rsi_len))
    k, d = stoch_rsi_kd(close, rsi_len=rsi_len, k_period=srsi_k, d_period=srsi_d, slowing=srsi_slow)
    k, d = align(k), align(d)
    (srsi_k_xup_d, srsi_k_xdn_d, srsi_k_xup_80, srsi_k_xdn_20,
     srsi_k_xup_d_50, srsi_k_xdn_d_50) = stoch_rsi_event_features(k, d, weight=srsi_w)
    di_plus, di_minus, adx = dmi_wilders(df, length=14)
    di_plus, di_minus, adx = align(di_plus), align(di_minus), align(adx)
    tod = align(time_of_day_feature(df.index, tz))
    ones = pd.Series(1.0, index=idx)

    cols = [
        ("ema5", ema5),("ema20", ema20),("ema50", ema50),("ema200", ema200),
        ("bb_pb", pb),("bb_bw", bw),
        ("atr", atr),
        ("rsi", rsi),
        ("srsi_k_xup_d", srsi_k_xup_d),("srsi_k_xdn_d", srsi_k_xdn_d),
        ("srsi_k_xup_80", srsi_k_xup_80),("srsi_k_xdn_20", srsi_k_xdn_20),
        ("srsi_k_xup_d_50", srsi_k_xup_d_50),("srsi_k_xdn_d_50", srsi_k_xdn_d_50),
        ("dmi_di_plus", di_plus),("dmi_di_minus", di_minus),("dmi_adx", adx),
    ]

    if include_regime:
        idx_tz = idx.tz_convert(tz)
        wd_frac = pd.Series(idx_tz.weekday/4.0, index=idx)
        m_sin = pd.Series(np.sin(2*np.pi*(idx_tz.month/12.0)), index=idx)
        m_cos = pd.Series(np.cos(2*np.pi*(idx_tz.month/12.0)), index=idx)
        vix_series = vix_series if vix_series is not None else load_vix_series(days=180, tz=tz)
        r_ids = regime_ids_from_flags(idx, close, rets, vix_series)
        bull = pd.Series((r_ids//12).astype(float), index=idx)
        rvq = ((r_ids%12)//3)+1
        rv_b = [pd.Series((rvq==q).astype(float), index=idx) for q in [1,2,3,4]]
        vb = (r_ids%3)+1
        vix_b = [pd.Series((vb==k).astype(float), index=idx) for k in [1,2,3]]
        vix_lvl = (load_vix_series(days=180, tz=tz).reindex(idx.date, method="ffill")
                   if not load_vix_series(days=180, tz=tz).empty else pd.Series(20, index=idx.date))
        vix_lvl = vix_lvl.reindex(idx.date, method="ffill").values
        vix_lvl = pd.Series(vix_lvl, index=idx)

        cols += [
            ("bull", bull),
            ("rv_q1", rv_b[0]),("rv_q2", rv_b[1]),("rv_q3", rv_b[2]),("rv_q4", rv_b[3]),
            ("vix_level", vix_lvl),
            ("vix_b1", vix_b[0]),("vix_b2", vix_b[1]),("vix_b3", vix_b[2]),
            ("wd_frac", wd_frac),("month_sin", m_sin),("month_cos", m_cos),
            ("fomc_flag", pd.Series(0.0, index=idx)),
            ("earnings_flag", pd.Series(0.0, index=idx)),
        ]

    cols += [("tod", tod), ("ones", ones)]
    F_core = pd.concat([s.rename(n) for (n,s) in cols], axis=1).fillna(0.0)
    return rets.values.astype(np.float32), F_core.values.astype(np.float32), list(F_core.columns), idx

def make_windows_no_cross(dt_index, arr_2d, L, tz="America/New_York"):
    out, idx_pairs = [], []
    if len(arr_2d) <= L: return np.zeros((0,L)+arr_2d.shape[1:], dtype=np.float32), idx_pairs
    day = pd.Series(dt_index.tz_convert(tz).date, index=dt_index)
    groups = day.groupby(day).groups
    for _, pos in groups.items():
        pos = list(pos); start = min(pos); end = max(pos)+1
        A = arr_2d[start:end]
        for i in range(0, len(A)-L):
            out.append(A[i:i+L]); idx_pairs.append((start+i, start+i+L))
    return np.asarray(out, dtype=np.float32), idx_pairs

# -------------------- Build conditioning (sidecar-aware) --------------------
def build_condition_windows_from_alpaca(tickers, days, seq_len, tz, sidecar):
    C_list, X_list, stats, meta = [], [], {}, []

    want_core = sidecar.get("core_feat_names") if sidecar else None
    use_tic = bool(sidecar and sidecar.get("use_ticker_embedding"))
    use_sec = bool(sidecar and sidecar.get("use_sector_embedding"))
    use_reg = bool(sidecar and sidecar.get("use_regime_embedding"))
    tic_block = sidecar.get("ticker_embedding") if sidecar else None
    sec_block = sidecar.get("sector_embedding") if sidecar else None
    reg_block = sidecar.get("regime_embedding") if sidecar else None

    vix_series = load_vix_series(days=180, tz=tz) if ((want_core and "vix_level" in want_core) or use_reg) else None

    for t in tickers:
        df = alpaca_df(t, days=days, tz=tz, rth_only=ALPACA_RTH_ONLY)
        rets, F_core, have_names, idx = build_features_from_df(
            df, tz=tz, include_regime=(want_core is not None and any(k in want_core for k in ["bull","rv_q1","vix_level","wd_frac"]))
        )

        core_names = want_core if want_core else have_names
        if want_core: F_core = reorder_or_pad_features(F_core, have_names, want_core)

        # per-ticker stats (sidecar preferred)
        if sidecar and "ret_stats" in sidecar and t in sidecar["ret_stats"]:
            rm = float(sidecar["ret_stats"][t]["mean"]); rs = float(sidecar["ret_stats"][t]["std"])
        else:
            _, rm, rs = zscore_np(rets)

        if sidecar and "feat_stats" in sidecar and t in sidecar["feat_stats"]:
            Fz = stats_apply_per_ticker(F_core, sidecar["feat_stats"][t])
        else:
            Fz = np.empty_like(F_core, dtype=np.float32)
            for j in range(F_core.shape[1]):
                m = float(np.nanmean(F_core[:,j])); s = float(np.nanstd(F_core[:,j])+1e-8)
                Fz[:,j] = (F_core[:,j] - m) / (s + 1e-8)

        Xz = ((rets - rm) / (rs + 1e-8)).reshape(-1,1).astype(np.float32)

        # windows
        Xw, seq_idx = make_windows_no_cross(idx, Xz, seq_len, tz=tz)
        Fw, _       = make_windows_no_cross(idx, Fz, seq_len, tz=tz)
        if len(Xw)==0: continue

        # embeddings expected by G
        extra_blocks = []

        if use_tic and tic_block:
            tic_list = tic_block["tickers"]; tic_dim = int(tic_block["dim"])
            tic_mat = np.array(tic_block["matrix"], dtype=np.float32)
            if t in tic_list: emb = tic_mat[tic_list.index(t)]
            else: emb = np.zeros((tic_dim,), dtype=np.float32)
            extra_blocks.append(np.tile(emb, (len(Xw), seq_len, 1)))

        if use_sec and sec_block:
            sec_dim = int(sec_block["dim"])
            sec_mat = np.array(sec_block["matrix"], dtype=np.float32)
            sectors = sec_block["sectors"]; t2s = sec_block.get("ticker_to_sector", {})
            sec_name = t2s.get(t, "FAKE")
            if sec_name in sectors: emb = sec_mat[sectors.index(sec_name)]
            else: emb = np.zeros((sec_dim,), dtype=np.float32)
            extra_blocks.append(np.tile(emb, (len(Xw), seq_len, 1)))

        if use_reg and reg_block:
            reg_dim = int(reg_block["dim"]); reg_mat = np.array(reg_block["matrix"], dtype=np.float32)
            close = df["Close"].astype(float)
            r_ids_full = regime_ids_from_flags(idx, close, rets, vix_series)
            reg_blocks = []
            for s,e in seq_idx:
                ids = np.clip(r_ids_full[s:e], 0, reg_mat.shape[0]-1)
                reg_blocks.append(reg_mat[ids])
            extra_blocks.append(np.array(reg_blocks, dtype=np.float32))

        # is_fake channel (real windows -> 0)
        is_fake = np.zeros((len(Xw), seq_len, 1), dtype=np.float32)

        C_part = np.concatenate(([Fw] + extra_blocks + [is_fake]), axis=-1)
        C_list.append(C_part); X_list.append(Xw)
        stats[t] = {"ret_mean": float(rm), "ret_std": float(rs), "last_price": float(df["Close"].iloc[-1])}
        meta.append({"ticker": t, "last_price": float(df["Close"].iloc[-1])})

    if not C_list:
        raise RuntimeError("No conditioning windows built from Alpaca data.")

    C_all = np.concatenate(C_list, axis=0)
    X_real= np.concatenate(X_list, axis=0)

    # Match cond_dim if specified
    if sidecar and "cond_dim" in sidecar:
        want = int(sidecar["cond_dim"]); have = C_all.shape[-1]
        if have < want:
            pad = np.zeros((C_all.shape[0], C_all.shape[1], want-have), dtype=np.float32)
            C_all = np.concatenate([C_all, pad], axis=-1)
        elif have > want:
            C_all = C_all[:, :, :want]
    return C_all, X_real, stats, meta

# -------------------- Generator I/O --------------------
def load_generator(gan_out_dir):
    paths = [os.path.join(gan_out_dir, "G_best.keras"),
             os.path.join(gan_out_dir, "G_intraday.keras"),
             os.path.join(gan_out_dir, "G_latest.keras")]
    for p in paths:
        if os.path.exists(p):
            try:
                G = tf.keras.models.load_model(p, compile=False)
                print(f"[G] Loaded {p}")
                return G
            except Exception:
                continue
    raise FileNotFoundError("No generator model found in GAN output dir.")

def infer_shapes_from_G(G, user_seq_len=None):
    c_shape = G.inputs[1].shape
    seq_len = int(c_shape[1]) if c_shape[1] is not None else (user_seq_len or 96)
    cond_dim = int(c_shape[-1])
    z_dim = int(G.inputs[0].shape[-1])
    return seq_len, cond_dim, z_dim

# -------------------- Episode providers --------------------
class GANEpisodeProvider:
    def __init__(self, G, cond_windows, z_dim, seq_len, gan_mix=1.0, real_windows=None, seed=0):
        self.G = G; self.C = cond_windows; self.z_dim = z_dim
        self.seq_len = seq_len; self.rng = np.random.RandomState(seed)
        self.gan_mix = float(np.clip(gan_mix, 0.0, 1.0)); self.real_windows = real_windows
    def sample_episode(self):
        use_gan = (self.rng.rand() < self.gan_mix) or (self.real_windows is None)
        if use_gan:
            i = self.rng.randint(0, len(self.C))
            c = self.C[i:i+1]  # (1,L,F)
            z = self.rng.normal(0,1,(1, self.z_dim)).astype(np.float32)
            fake = self.G.predict([z, c], verbose=0).squeeze(axis=0)  # (L,1)
            return fake.astype(np.float32)
        else:
            j = self.rng.randint(0, len(self.real_windows))
            return self.real_windows[j].astype(np.float32)

class RealEpisodeProvider:
    def __init__(self, real_windows, seed=0):
        self.X = real_windows; self.rng = np.random.RandomState(seed)
    def sample_episode(self):
        i = self.rng.randint(0, len(self.X))
        return self.X[i].astype(np.float32)

# -------------------- Trading env --------------------
class TradingEnv:
    """Episodic env over normalized returns; actions {-1,0,+1}; reward = log equity change."""
    def __init__(self, episode_provider, seq_len=96, tcost=1e-4, pos_penalty=0.0,
                 window=16, start_cash=1000.0, seed=0):
        self.provider = episode_provider; self.seq_len = seq_len
        self.tcost = tcost; self.pos_penalty = pos_penalty
        self.window = window; self.start_cash = start_cash
        self.rng = np.random.RandomState(seed)
        self.action_space_n = 3; self.observation_space_n = window + 3 + 1
        self.reset()
    def reset(self):
        self.rets = self.provider.sample_episode().reshape(-1)  # (L,)
        self.t = 0; self.pos = 0; self.prev_pos = 0
        self.equity = self.start_cash; self.last_equity = self.equity
        self.window_buf = [0.0]*self.window
        return self._obs()
    def _obs(self):
        w = np.array(self.window_buf, dtype=np.float32)
        pos_oh = np.zeros(3, dtype=np.float32); pos_oh[self.pos+1] = 1.0
        cash_frac = np.array([self.equity / self.start_cash], dtype=np.float32)
        return np.concatenate([w, pos_oh, cash_frac], axis=0)
    def step(self, action):
        action = int(np.clip(action, 0, 2))
        self.prev_pos = self.pos; self.pos = action - 1
        r_t = float(self.rets[self.t])
        pos_change = abs(self.pos - self.prev_pos)
        tcost_loss = self.tcost * pos_change
        pos_pen = self.pos_penalty * abs(self.pos)
        growth = (self.pos * r_t) - tcost_loss - pos_pen
        self.equity *= (1.0 + growth)
        reward = np.log(self.equity / self.last_equity + 1e-12)
        self.last_equity = self.equity
        self.window_buf = (self.window_buf + [r_t])[1:]
        self.t += 1
        done = (self.t >= self.seq_len)
        return self._obs(), float(reward), done, {"equity": self.equity, "r_t": r_t, "pos": self.pos}

# -------------------- DQN --------------------
def build_qnet(obs_dim, n_actions):
    x_in = Input(shape=(obs_dim,))
    x = Dense(128, activation="relu")(x_in)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    q = Dense(n_actions, activation=None)(x)
    return Model(x_in, q)

class Replay:
    def __init__(self, cap=50000):
        self.cap = cap; self.buf = []; self.pos = 0
    def add(self, s, a, r, s2, d):
        if len(self.buf) < self.cap: self.buf.append(None)
        self.buf[self.pos] = (s,a,r,s2,d); self.pos = (self.pos+1) % self.cap
    def sample(self, bs):
        idx = np.random.randint(0, len(self.buf), size=bs)
        batch = [self.buf[i] for i in idx]
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s.astype(np.float32), a.astype(np.int32), r.astype(np.float32), s2.astype(np.float32), d.astype(np.float32)
    def __len__(self): return len(self.buf)

# -------------------- Evaluation + plotting --------------------
def eval_on_alpaca_and_plot(qnet, ticker, days, tz, tcost_bps, pos_pen_bps, seed, out_dir,
                            save_trades_csv=True, save_plot=True):
    """
    Runs policy on real Alpaca 1m data. Saves:
    - trades_<ticker>.csv (timestamp, close, action, position, equity)
    - trade_plot_<ticker>.png (price with buy/sell markers)
    """
    df = alpaca_df(ticker, days=days, tz=tz, rth_only=ALPACA_RTH_ONLY)
    close = df["Close"].astype(float)
    rets = to_returns(close).values.astype(np.float32)
    idx = to_returns(close).index  # aligns with rets
    seq_len = min(1000, len(rets)-1 if len(rets)>1 else len(rets))

    # Build a provider with a single sequence and keep index to map steps->timestamps
    real_windows = rets[:seq_len].reshape(1, -1, 1)
    provider = RealEpisodeProvider(real_windows, seed=seed)
    env = TradingEnv(provider, seq_len=seq_len,
                     tcost=tcost_bps*1e-4, pos_penalty=pos_pen_bps*1e-4,
                     window=16, start_cash=1000.0, seed=seed)

    s = env.reset()
    equity_curve=[env.equity]; positions=[0]; actions=[1]  # 0=short,1=flat,2=long; start flat
    # keep track at each step t the timestamp (rets index t)
    times = [idx[0]]
    prices = [close.loc[idx[0]]]

    done=False; t=0
    while not done:
        qv = qnet.predict(s[None,:], verbose=0)[0]
        a = int(np.argmax(qv))
        s, r, done, info = env.step(a)
        t += 1
        # map to timestamp: rets[t] corresponds to price at idx[t]
        ts = idx[min(t, len(idx)-1)]
        times.append(ts)
        prices.append(close.loc[ts])
        equity_curve.append(info["equity"])
        actions.append(a)
        positions.append(info["pos"])

    # Save trades csv (only when position changes -> entry/exit)
    trades = []
    prev_pos = 0
    for i in range(1, len(positions)):
        if positions[i] != prev_pos:
            trades.append({
                "timestamp": times[i],
                "close": prices[i],
                "action": actions[i],         # raw action 0/1/2
                "position": positions[i],     # -1/0/+1
                "equity": equity_curve[i]
            })
            prev_pos = positions[i]
    trades_df = pd.DataFrame(trades)

    os.makedirs(out_dir, exist_ok=True)
    if save_trades_csv:
        trades_df.to_csv(os.path.join(out_dir, f"trades_{ticker}.csv"), index=False)

    # Plot price with buy/sell markers
    if save_plot:
        plt.figure(figsize=(11,5))
        plt.plot(times, prices, label=f"{ticker} Close", linewidth=1.2)
        # mark transitions
        for row in trades:
            if row["position"] == 1 and (row["action"] == 2):  # entered long
                plt.scatter(row["timestamp"], row["close"], marker="^", s=70, label="Buy (long)", zorder=3)
            elif row["position"] == -1 and (row["action"] == 0):  # entered short
                plt.scatter(row["timestamp"], row["close"], marker="v", s=70, label="Sell (short)", zorder=3)
            elif row["position"] == 0:
                plt.scatter(row["timestamp"], row["close"], marker="o", s=45, label="Flat", zorder=3)
        # avoid duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), loc="best")
        plt.title(f"Trades overlay — {ticker} ({days}d, 1m)")
        plt.xlabel("Time"); plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trade_plot_{ticker}.png"), dpi=150)
        plt.close()

    final_ret = (equity_curve[-1]/equity_curve[0]-1)*100.0
    print(f"[EVAL] {ticker}: Final equity ${equity_curve[-1]:.2f} | Return {final_ret:.2f}%")
    return np.array(equity_curve), trades_df

# -------------------- Main --------------------
def main():
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
    out_dir = OUT_DIR or os.path.join(GAN_OUT_DIR, "rl_agent"); os.makedirs(out_dir, exist_ok=True)

    # Load generator & sidecar
    G = load_generator(GAN_OUT_DIR)
    seq_len_G, cond_dim, z_dim = infer_shapes_from_G(G, user_seq_len=None)
    sidecar = load_sidecar_meta(GAN_OUT_DIR)

    # Build Alpaca conditioning windows (sidecar-aware)
    C_for_G, X_real_all, stats_map, _meta = build_condition_windows_from_alpaca(
        tickers=ALPACA_TICKERS,
        days=ALPACA_DAYS,
        seq_len=seq_len_G,
        tz=ALPACA_TZ,
        sidecar=sidecar
    )

    # Episode provider
    if USE_GAN_SAMPLER:
        provider = GANEpisodeProvider(G, C_for_G, z_dim, C_for_G.shape[1],
                                      gan_mix=GAN_MIX, real_windows=X_real_all, seed=SEED)
    else:
        provider = RealEpisodeProvider(X_real_all, seed=SEED)

    # Env & agent
    env = TradingEnv(provider, seq_len=C_for_G.shape[1],
                     tcost=TCOST_BPS*1e-4, pos_penalty=POS_PENALTY_BPS*1e-4,
                     window=16, start_cash=1000.0, seed=SEED)
    obs_dim = env.observation_space_n; n_actions = env.action_space_n
    qnet = build_qnet(obs_dim, n_actions)
    tgt  = build_qnet(obs_dim, n_actions); tgt.set_weights(qnet.get_weights())
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    replay = Replay(REPLAY_SIZE)

    def eps_at(t):
        if t >= EPSILON_DECAY_STEPS: return EPSILON_END
        return EPSILON_START + (EPSILON_END-EPSILON_START)*(t/EPSILON_DECAY_STEPS)

    @tf.function
    def train_step(s,a,r,s2,d, gamma):
        with tf.GradientTape() as tape:
            q = qnet(s)
            qa = tf.gather(q, a[:,None], axis=1, batch_dims=1)[:,0]
            q2 = tgt(s2); maxq2 = tf.reduce_max(q2, axis=1)
            y = r + gamma*(1.0 - d)*maxq2
            loss = tf.reduce_mean(tf.square(qa - tf.stop_gradient(y)))
        grads = tape.gradient(loss, qnet.trainable_variables)
        opt.apply_gradients(zip(grads, qnet.trainable_variables))
        return loss

    # Train
    print("Training DQN…")
    step, ep = 0, 0; rewards=[]
    while ep < EPISODES and step < MAX_STEPS:
        s = env.reset(); done=False; ep_r=0.0
        while not done:
            e = eps_at(step)
            if np.random.rand() < e: a = np.random.randint(0, n_actions)
            else: a = int(np.argmax(qnet.predict(s[None,:], verbose=0)[0]))
            s2, r, done, info = env.step(a)
            replay.add(s,a,r,s2,float(done))
            s = s2; ep_r += r; step += 1

            if len(replay) >= BATCH_SIZE:
                sb,ab,rb,s2b,db = replay.sample(BATCH_SIZE)
                train_step(sb,ab,rb,s2b,db, tf.constant(GAMMA, dtype=tf.float32))

            if step % TARGET_UPDATE_EVERY == 0:
                tgt.set_weights(qnet.get_weights())
            if step >= MAX_STEPS: break
        rewards.append(ep_r); ep += 1
        if ep % 20 == 0:
            print(f"Episode {ep}/{EPISODES} | epR={ep_r:.4f} | equity=${info['equity']:.2f} | eps={e:.3f}")

    qnet.save(os.path.join(out_dir, "dqn_policy.keras"))
    pd.Series(rewards).rolling(10).mean().to_csv(os.path.join(out_dir,"rewards_ma10.csv"), index=False)
    plt.figure(figsize=(8,4)); plt.plot(pd.Series(rewards).rolling(10).mean())
    plt.title("DQN episode reward (10-ep MA)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rewards.png"), dpi=140); plt.close()

    # Evaluate on real Alpaca 1m and save trade plot + csv
    eq_curve, trades_df = eval_on_alpaca_and_plot(
        qnet=qnet,
        ticker=EVAL_TICKER,
        days=EVAL_DAYS,
        tz=ALPACA_TZ,
        tcost_bps=TCOST_BPS,
        pos_pen_bps=POS_PENALTY_BPS,
        seed=SEED,
        out_dir=out_dir,
        save_trades_csv=SAVE_TRADES_CSV,
        save_plot=SAVE_TRADE_PLOT
    )

    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
