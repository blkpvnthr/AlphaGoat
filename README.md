# 🐐 AlphaGoat

**AlphaGoat** is a full-stack **quant research and trading sandbox** that fuses  
**Generative Adversarial Networks (GANs)** with **Reinforcement Learning (RL)**  
to create realistic intraday market data and train trading agents end-to-end.

---

## ✨ Features

- **Market Simulator (GAN)**  
  - Conditional Wasserstein GAN with rich technical-indicator conditioning  
  - Generates 1-minute synthetic price paths that mimic real-market regimes  

- **Real-Time Data Integration**  
  - Pulls live 1-minute bars from **Alpaca Market Data v2**  
  - Auto-builds indicator features (EMA 5/20/50/200, Bollinger Bands, ATR,  
    RSI, StochRSI event flags, DMI, volatility regimes, calendar signals)

- **Reinforcement-Learning Trader**  
  - Deep Q-Network (DQN) agent starts with **$1,000** and trades **long / flat / short**  
  - Optimizes log-equity growth with transaction-cost and position penalties  
  - Optionally trains on a mix of GAN-generated and real Alpaca windows

- **Evaluation & Visualization**  
  - Reconstructs price paths from normalized returns  
  - Saves equity curves, trade logs (`trades_<ticker>.csv`)  
  - Generates a **buy/sell overlay plot** on real price data

---

## 🚀 Quick Start

1. **Clone & install requirements**

 ```bash
    git clone https://github.com/blkpvnthr/AlphaGoat.git
    cd AlphaGoat
    pip install -r file.txt
 ```
 > These versions are broad enough to stay compatible with current macOS/Linux setups while ensuring TensorFlow ≥ 2.15 for tf.keras features.
<\br> 

2. **Create an Alpaca account & API keys**

AlphaGoat needs live 1-minute market data. Follow these steps:

Sign up
Go to <a href="https://alpaca.markets" target="_blank">https://alpaca.markets</a> and create a free account.
> (choose Paper Trading if you don’t plan to trade real money).

- Verify your email & log in.
- Generate API keys
- Click API Keys → Generate Key
- Copy API Key ID and Secret Key.
- Store the keys as environment variables (recommended)
  
```bash
  export ALPACA_API_KEY_ID="your_key_id"
  export ALPACA_API_SECRET_KEY="your_secret_key"
```

> On macOS you can add these to ~/.zshrc or ~/.bash_profile to persist.
<\br>

3. **Train & evaluate**

   Train the model

   ```bash
   python gang-gang.py
   ```

   Backtest using RL agent

   ```bash
   python rl_trader.py
   ```

   - Trains a DQN agent on GAN-sampled and/or real Alpaca 1-minute data
   - Evaluates on live Alpaca data and saves trade plots in rl_agent/
  
---

## 📂 Repository Layout
<pre> 
AlphaGoat/
├─ rl_trader.py        # main training & evaluation script
├─ gang-gang.py        # original GAN training script
├─ models/             # trained generator & RL policy checkpoints
├─ output/             # equity curves, trade logs, plots
└─ file.txt
</pre>

---

## ⚙️ Configuration

All runtime settings live at the top of rl_trader.py as plain Python variables
(e.g. tickers, training episodes, cost parameters).
No command-line arguments required—edit and run.

---

## 🧠 Roadmap

- Transformer/TCN generator option
- Diffusion-based price generator
- Additional RL algorithms (PPO, SAC)
- Automated hyper-parameter sweeps
  
 ---

## 🪪 License

MIT License © 2025 BLKPVNTHR

---

## Disclaimer

> This project is for research and educational purposes only.
> It is not investment advice and should not be used for live trading.