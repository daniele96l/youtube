import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import re

# =========================================================
# Parametri principali
# =========================================================
YEARS = 10
SIMS = 1000
ANNUAL_INVEST = 12000.0
SEED = 42
RF = 0.0   # risk-free per Sharpe (R_to_V) in forma decimale (0.02 = 2%)

# Solo soglie Buy The Dip — include 0% (investe sempre tutto). Nessuna baseline.
DIP_THRESHOLDS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]

# Esempi per i grafici BTD (griglia)
EXAMPLES_PER_CASE = 4   # quanti esempi per soglia
EXAMPLE_GRID_COLS = 2   # colonne della griglia (righe calcolate automaticamente)

# Pannelli principali: % investita fissa
FIXED_PCTS = [0.50, 0.70, 0.90]

T = YEARS * 12
rng = np.random.default_rng(SEED)
random.seed(SEED)

# =========================================================
# Dati S&P 500: rendimenti mensili storici
# =========================================================
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")
if sp500_data.empty:
    raise ValueError("Impossibile scaricare i dati S&P 500")

sp500_close = sp500_data["Close"]
# Ensure the index is datetime and fix the deprecated 'M' to 'ME'
monthly_returns_hist = sp500_close.resample("ME").last().pct_change().dropna().values
if len(monthly_returns_hist) < 12:
    raise ValueError("Storico mensile insufficiente per la simulazione.")

# =========================================================
# Campionamento Monte Carlo UNA VOLTA (stessa base R per tutti i casi)
# =========================================================
R_base = rng.choice(monthly_returns_hist, size=(SIMS, T), replace=True)

# Indice di mercato cumulato (parte da 1.0) per ogni simulazione
MARKET_LEVELS = np.cumprod(1.0 + R_base, axis=1)

# =========================================================
# Simulatore (vettorizzato su SIMS)
# =========================================================
def simulate_pct_with_strategy(pct: float, annual_invest: float, R: np.ndarray, dip_threshold: float):
    """
    pct: quota del contributo mensile destinata all'investimento continuo (PAC).
    dip_threshold:
      - 0.0   -> Caso speciale "investe sempre tutto": 100% dei flussi va subito investito, nessun cash.
      - >0.0  -> Regola BTD classica: accumula cash e deploya quando drawdown raggiunge la soglia.
    Nota: non esiste più la 'baseline' (dip_threshold=None).
    """
    sims, horizon = R.shape

    # Caso speciale: BTD 0% = investe sempre tutto. Ignoriamo pct e forziamo 100% investito.
    if dip_threshold <= 0.0:
        c_inv = (annual_invest) / 12.0
        c_cash = 0.0
    else:
        c_inv = (annual_invest * pct) / 12.0
        c_cash = (annual_invest * (1.0 - pct)) / 12.0

    invested = np.zeros((sims, horizon), dtype=float)
    cash     = np.zeros((sims, horizon), dtype=float)
    trigger  = np.zeros((sims, horizon), dtype=bool)
    deploy_amounts = np.zeros((sims, horizon), dtype=float)

    invested_cur = np.zeros(sims, dtype=float)
    cash_cur     = np.zeros(sims, dtype=float)

    level_pre = np.ones((sims,), dtype=float)
    roll_max  = np.ones((sims,), dtype=float)
    prev_dd   = np.zeros((sims,), dtype=float)

    for t in range(horizon):
        drawdown = 1.0 - (level_pre / roll_max)

        # Regola BTD: attiva solo se threshold > 0
        if dip_threshold > 0.0:
            mask = (prev_dd < dip_threshold) & (drawdown >= dip_threshold)
            if np.any(mask):
                deploy_amounts[mask, t] = cash_cur[mask]     # deploy = trasferimento interno
                invested_cur[mask] += cash_cur[mask]
                trigger[mask, t] = True
                cash_cur[mask] = 0.0

        prev_dd = drawdown

        # contributi del mese (flussi esterni)
        invested_cur += c_inv
        cash_cur     += c_cash

        # rendimento del mese (solo sulla parte investita)
        invested_cur *= (1.0 + R[:, t])

        invested[:, t] = invested_cur
        cash[:, t]     = cash_cur

        level_pre *= (1.0 + R[:, t])
        roll_max = np.maximum(roll_max, level_pre)

    total = invested + cash
    return invested, cash, total, trigger, deploy_amounts

# =========================================================
# Wrapper mediane + paths (solo BTD)
# =========================================================
def summarize_over_time_fixed_pct(fixed_pct: float, thresholds: list[float], R: np.ndarray):
    """
    Mediane nel tempo per le sole soglie BTD (0%, 5%, 10%, ...), a % investita fissa.
    """
    out = {}
    for thr in thresholds:
        inv, cash, tot, _, _ = simulate_pct_with_strategy(fixed_pct, ANNUAL_INVEST, R, dip_threshold=thr)
        out[f"BTD {int(thr*100)}%"] = {
            "invested_p50": np.percentile(inv, 50, axis=0),
            "cash_p50":     np.percentile(cash, 50, axis=0),
            "total_p50":    np.percentile(tot, 50, axis=0),
        }
    return out

def summarize_with_paths_fixed_pct(fixed_pct: float, thresholds: list[float], R: np.ndarray):
    """
    Per esempi e metriche: restituisce paths e trigger per ogni soglia.
    """
    invested_paths = {}
    cash_paths = {}
    total_paths = {}
    trigger_paths = {}
    deploy_paths = {}
    for thr in thresholds:
        inv, cash, tot, trig, dep = simulate_pct_with_strategy(fixed_pct, ANNUAL_INVEST, R, dip_threshold=thr)
        key = f"BTD {int(thr*100)}%"
        invested_paths[key] = inv
        cash_paths[key] = cash
        total_paths[key] = tot
        trigger_paths[key] = trig
        deploy_paths[key] = dep
    return invested_paths, cash_paths, total_paths, trigger_paths, deploy_paths

# =========================================================
# Metriche TWR (netto dei depositi) su portafoglio totale
# =========================================================
def compute_final_and_twr_metrics(total_paths: np.ndarray,
                                  invested_paths: np.ndarray,
                                  cash_paths: np.ndarray,
                                  R: np.ndarray):
    """
    Calcola:
      - final_vals: capitale finale (investito + cash)
      - twr_monthly: rendimenti mensili TWR del portafoglio (flussi esterni rimossi)
      - R_ann_twr: rendimento annuo TWR (geometrico)
      - Vol_ann_twr: volatilità annua dei rendimenti TWR (std mensile * sqrt(12))
    """
    final_vals = total_paths[:, -1]
    base_t = invested_paths / (1.0 + R) + cash_paths        # valore prima del rendimento (post-flussi esterni)
    total_end_t = invested_paths + cash_paths               # valore a fine mese

    with np.errstate(divide='ignore', invalid='ignore'):
        twr_monthly = (total_end_t - base_t) / base_t

    vol_ann_twr = np.nanstd(twr_monthly, axis=1) * np.sqrt(12)

    twr_monthly_clean = np.where(np.isnan(twr_monthly), 0.0, twr_monthly)
    growth = np.prod(1.0 + twr_monthly_clean, axis=1)
    R_ann_twr = growth**(12.0 / twr_monthly_clean.shape[1]) - 1.0

    return final_vals, vol_ann_twr, R_ann_twr

def metrics_for_fixed_pct_across_thresholds(fixed_pct: float, thresholds: list[float], R: np.ndarray):
    """
    Calcola TWR (rendimento, volatilità) e Sharpe per le sole soglie BTD, alla % investita fissa.
    """
    labels = []; vols = []; rets = []; sharpes = []; finals = []

    for thr in thresholds:
        inv, cash, tot, _, _ = simulate_pct_with_strategy(fixed_pct, ANNUAL_INVEST, R, dip_threshold=thr)
        f, v_twr, r_twr = compute_final_and_twr_metrics(tot, inv, cash, R)
        r50 = np.percentile(r_twr, 50)
        v50 = np.percentile(v_twr, 50)
        sh  = (r50 - RF) / v50 if v50 > 0 else np.nan
        labels.append(f"BTD {int(thr*100)}%")
        vols.append(v50 * 100); rets.append(r50 * 100); sharpes.append(sh)
        finals.append(np.percentile(f, 50))

    df = pd.DataFrame({
        "Scenario": labels,
        "R_%": rets,
        "V_%": vols,
        "Sharpe_TWR": sharpes,
        "Capitale_finale_p50": finals,
    })
    # Ordine naturale per soglia crescente
    order = [f"BTD {int(x*100)}%" for x in thresholds]
    df["Scenario"] = pd.Categorical(df["Scenario"], categories=order, ordered=True)
    return df.sort_values("Scenario").reset_index(drop=True)

# =========================================================
# Plot helpers
# =========================================================
months = np.arange(1, T+1)

def _sort_btd_labels(labels):
    """
    Ordina etichette tipo 'BTD 0%', 'BTD 5%', ... per valore numerico.
    """
    def extract_num(s):
        m = re.search(r'(\d+)', s)
        return int(m.group(1)) if m else 10**9
    return sorted(labels, key=extract_num)

def plot_total_over_time_fixed_pct(results_by_thr: dict, fixed_pct: float):
    plt.figure(figsize=(12,7))
    for lbl in _sort_btd_labels(list(results_by_thr.keys())):
        d = results_by_thr[lbl]
        plt.plot(months, d["total_p50"], label=lbl, linewidth=1.8)
    plt.title(f"Totale (Investito + Liquidità) — Mediana ({int(fixed_pct*100)}% investito)")
    plt.xlabel("Mese"); plt.ylabel("Valore (€)")
    plt.grid(True, alpha=0.3); plt.legend(ncol=3); plt.tight_layout(); plt.show()

def plot_invested_vs_cash_grid_fixed_pct(results_by_thr: dict, fixed_pct: float):
    labels = _sort_btd_labels(list(results_by_thr.keys()))
    n = len(labels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.7*rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1, cols)
    for i, lbl in enumerate(labels):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        d = results_by_thr[lbl]
        ax.plot(months, d["invested_p50"], label="Investito (med)", linewidth=1.8)
        ax.plot(months, d["cash_p50"],     label="Risparmiati (med)", linewidth=1.8, linestyle="--")
        ax.set_title(lbl); ax.grid(True, alpha=0.3)
        if r == rows-1: ax.set_xlabel("Mese")
        if c == 0: ax.set_ylabel("€")
    handles, lab = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=2)
    fig.suptitle(f"Investito vs Cash (mediana) — {int(fixed_pct*100)}% investito", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.show()

# ====== Esempi BTD in GRIGLIA (asse doppio: € a sinistra, indice a destra) ======
def plot_btd_examples_grid(
    pct_label,
    cash_paths,
    trigger_paths,
    deploy_amounts_paths,
    market_levels,                  # matrice (SIMS, T) dell'indice di mercato (base 1.0)
    threshold,
    n_examples=4,
    cols=2
):
    """
    Mostra esempi in griglia con due assi Y:
      - Asse sinistro (€, linea tratteggiata): 'Soldi risparmiati'
      - Asse destro (Indice base 100): 'Mercato azionario'
    Le linee verticali indicano i mesi di trigger (deploy del cash).
    """
    sim_count = cash_paths.shape[0]
    n_examples = min(n_examples, sim_count)
    sample_idx = random.sample(range(sim_count), n_examples)

    rows = math.ceil(n_examples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.0*cols, 3.2*rows), sharex=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for ax, idx in zip(axes_flat, sample_idx):
        cas = cash_paths[idx]                  # € (sinistra)
        mkt = market_levels[idx]               # indice (1.0, 1.1, ...)
        mkt_idx = 100.0 * (mkt / mkt[0])       # normalizza a 100

        # Asse destro per il mercato
        ax2 = ax.twinx()

        # Tracce
        ln1, = ax2.plot(months, mkt_idx, linewidth=1.6, label="Mercato azionario (indice=100)")
        ln2, = ax.plot(months, cas, linewidth=1.6, linestyle="--", label="Soldi risparmiati (€)")

        # Trigger BTD: linee verticali sull'asse sinistro (coprono tutta l'altezza dell'asse sinistro)
        trg = trigger_paths[idx]
        dep = deploy_amounts_paths[idx]
        trigger_months = months[trg]
        for tm in trigger_months:
            ax.axvline(tm, linewidth=0.8, alpha=0.2)

        # Scatter dei deploy (in €) sul sinistro
        if trigger_months.size > 0:
            ax.scatter(trigger_months, dep[trg], marker="o", s=20)

        # Etichette assi
        ax.set_ylabel("€ (cash)")
        ax2.set_ylabel("Indice (base=100)")

        # Griglia sul sinistro
        ax.grid(True, alpha=0.3)

    # Nascondi assi non usati
    for ax in axes_flat[n_examples:]:
        ax.axis("off")

    # Legenda combinata (sinistra+destra) e titolo
    # Prendiamo le handles da primo pannello valido
    first_ax = axes_flat[0]
    first_ax2 = first_ax.twinx()  # crea temporaneo solo per prelevare labels se servisse
    handles, labels = [], []
    # Recuperiamo dal primo pannello reale (quello dentro il loop ha ln1/ln2 locali)
    # Qui ricostruiamo le etichette per robustezza:
    handles = []
    labels = []
    handles.append(plt.Line2D([0], [0]))  # placeholder, verrà sostituito subito
    labels.append("Mercato azionario (indice=100)")
    handles[0] = plt.Line2D([0], [0])  # rebind per sicurezza

    # Meglio: ricava da un asse del loop (se esistono esempi)
    if len(axes_flat) > 0 and n_examples > 0:
        # Riplotta invisibile per ottenere handle/label in maniera pulita
        tmp_ax = axes_flat[0]
        tmp_ax2 = tmp_ax.twinx()
        h1, l1 = tmp_ax2.get_legend_handles_labels()
        h2, l2 = tmp_ax.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2

    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        f"Esempi Buy The Dip — {pct_label}, soglia {int(threshold*100)}%",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ====== Frontiere efficienti (discrete) ======
def _frontier_from_points(vols, rets):
    order = np.argsort(vols)
    vols_sorted = vols[order]; rets_sorted = rets[order]
    keep_idx = []; best_ret = -np.inf
    for i, (v, r) in enumerate(zip(vols_sorted, rets_sorted)):
        if r >= best_ret:
            keep_idx.append(order[i]); best_ret = r
    return np.array(keep_idx, dtype=int)

def _frontier_plot(ax, x_vols, y_rets, labels=None, title=""):
    idx_front = _frontier_from_points(x_vols, y_rets)
    f_order = np.argsort(x_vols[idx_front])
    vf = x_vols[idx_front][f_order]; rf = y_rets[idx_front][f_order]
    ax.scatter(x_vols, y_rets, s=36)
    if labels is not None:
        for (vx, ry, lab) in zip(x_vols, y_rets, labels):
            ax.annotate(lab, (vx, ry), xytext=(4,2), textcoords="offset points", fontsize=8)
    ax.plot(vf, rf, linewidth=2)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Vol ann (TWR, %)")
    ax.set_ylabel("R ann (TWR, %)")

# ====== Frontiere in griglia — per % investita fissa (solo BTD) ======
def plot_frontiers_grid_fixed_pcts(fixed_pcts: list[float], thresholds: list[float], R: np.ndarray, cols=2):
    n = len(fixed_pcts)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0*cols, 5.5*rows))
    axes = np.array(axes).reshape(-1)

    for i, fp in enumerate(fixed_pcts):
        vols = []
        rets = []
        labs = []
        for thr in thresholds:
            inv, cash, tot, _, _ = simulate_pct_with_strategy(fp, ANNUAL_INVEST, R, dip_threshold=thr)
            f, v_twr, r_twr = compute_final_and_twr_metrics(tot, inv, cash, R)
            vols.append(np.percentile(v_twr, 50) * 100)
            rets.append(np.percentile(r_twr, 50) * 100)
            labs.append(f"BTD {int(thr*100)}%")

        _frontier_plot(axes[i], np.array(vols), np.array(rets), labels=labs,
                       title=f"{int(fp*100)}% investito")

    # Nascondi assi extra
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Frontiere efficienti (TWR) — griglia per % investita fissa (solo BTD)", y=0.995)
    plt.tight_layout()
    plt.show()

# ====== Frontiere — vista “scenario = % investito” (solo BTD) ======
def plot_frontiers_grid_by_fixedpct(summary_df: pd.DataFrame, fixed_pcts: list[float], cols=3):
    scenarios = [f"{int(fp*100)}% investito" for fp in fixed_pcts]
    df = summary_df.copy()
    df["R_%"] = df["R_ann_medio_p50"] * 100
    df["V_%"] = df["Vol_ann_p50"] * 100

    n = len(scenarios) + 1  # +1 per "Totale"
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5*cols, 5.0*rows))
    axes = np.array(axes).reshape(-1)

    for i, scn in enumerate(scenarios):
        sub = df[df["Scenario"] == scn]
        vols = sub["V_%"].to_numpy()
        rets = sub["R_%"].to_numpy()
        labels = sub["Percentuale"].astype(str).to_numpy()  # "BTD xx%"
        _frontier_plot(axes[i], vols, rets, labels=labels, title=scn)

    # Pannello "Totale"
    all_vols = df["V_%"].to_numpy()
    all_rets = df["R_%"].to_numpy()
    all_labels = (df["Scenario"].astype(str) + " — " + df["Percentuale"].astype(str)).to_numpy()
    _frontier_plot(axes[len(scenarios)], all_vols, all_rets, labels=all_labels, title="Totale")

    # Nascondi eventuali assi in eccesso
    for j in range(len(scenarios)+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Frontiere efficienti (TWR) — per % investita fissa e totale (solo BTD)", y=0.995)
    plt.tight_layout()
    plt.show()

# =========================================================
# ========== MAIN ==========
# % investita FISSA, variabile = Soglia BTD (incluso 0% = investe sempre tutto)
# Nessuna baseline in nessun plot.
# =========================================================
SHOW_EXAMPLES = True  # mostra o meno gli esempi con trigger

all_metrics_rows = []

for fp in FIXED_PCTS:
    # Serie temporali (mediane) — solo BTD
    res_time = summarize_over_time_fixed_pct(fp, DIP_THRESHOLDS, R_base)
    plot_total_over_time_fixed_pct(res_time, fp)
    plot_invested_vs_cash_grid_fixed_pct(res_time, fp)

    # Metriche aggregate TWR vs soglia (solo BTD)
    df_thr = metrics_for_fixed_pct_across_thresholds(fp, DIP_THRESHOLDS, R_base)

    # Linee R/Vol + barre Sharpe
    x_labels = df_thr["Scenario"].tolist()
    x = np.arange(len(x_labels))

    plt.figure(figsize=(12,5.5))
    plt.plot(x, df_thr["R_%"], linewidth=2, label="R% (TWR)")
    plt.plot(x, df_thr["V_%"], linewidth=2, label="Vol% (TWR)")
    plt.xticks(x, x_labels, rotation=0)
    plt.title(f"{int(fp*100)}% investito — R% & Vol% (TWR) vs soglia BTD")
    plt.ylabel("%"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12,5))
    plt.bar(x, df_thr["Sharpe_TWR"])
    plt.xticks(x, x_labels, rotation=0)
    plt.title(f"{int(fp*100)}% investito — Sharpe (TWR) vs soglia BTD")
    plt.ylabel("Sharpe (TWR)"); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.show()

    # Righe metriche per frontiere (solo BTD)
    scn_label = f"{int(fp*100)}% investito"
    tmp_rows = []
    for thr in DIP_THRESHOLDS:
        inv, cash, tot, _, _ = simulate_pct_with_strategy(fp, ANNUAL_INVEST, R_base, dip_threshold=thr)
        finals, vols_twr, rets_twr = compute_final_and_twr_metrics(tot, inv, cash, R_base)
        r50 = np.percentile(rets_twr, 50)
        v50 = np.percentile(vols_twr, 50)
        sharpe = (r50 - RF) / v50 if v50 > 0 else np.nan
        tmp_rows.append({
            "Scenario": scn_label,                 # es. "50% investito"
            "Percentuale": f"BTD {int(thr*100)}%", # solo BTD
            "Capitale_finale_p05": np.percentile(finals, 5),
            "Capitale_finale_p50": np.percentile(finals, 50),
            "Capitale_finale_p95": np.percentile(finals, 95),
            "R_ann_medio_p50":     r50,
            "Vol_ann_p50":         v50,
            "R_to_V_p50":          sharpe
        })

    df_fp_metrics = pd.DataFrame(tmp_rows)
    all_metrics_rows.append(df_fp_metrics)

    # (Opzionale) Esempi “buy the dip” in GRIGLIA per ciascuna soglia — con asse doppio
    if SHOW_EXAMPLES:
        for thr in DIP_THRESHOLDS:
            inv, cash, tot, trig, dep = simulate_pct_with_strategy(fp, ANNUAL_INVEST, R_base, dip_threshold=thr)
            pct_lbl = f"{int(fp*100)}%"
            plot_btd_examples_grid(
                pct_label=pct_lbl,
                cash_paths=cash,
                trigger_paths=trig,
                deploy_amounts_paths=dep,
                market_levels=MARKET_LEVELS,   # indice di mercato
                threshold=thr,
                n_examples=EXAMPLES_PER_CASE,
                cols=EXAMPLE_GRID_COLS
            )

# A) Griglia delle frontiere per le % fisse (solo BTD)
plot_frontiers_grid_fixed_pcts(FIXED_PCTS, DIP_THRESHOLDS, R_base, cols=2)

# B) Frontiera "mega" su tutte le % fisse e soglie (Scenario = "% investito", solo BTD)
summary_df_new = pd.concat(all_metrics_rows, ignore_index=True)
scenario_order_new = [f"{int(fp*100)}% investito" for fp in FIXED_PCTS]
summary_df_new["Scenario"] = pd.Categorical(summary_df_new["Scenario"], categories=scenario_order_new, ordered=True)
thr_order = [f"BTD {int(x*100)}%" for x in DIP_THRESHOLDS]
summary_df_new["Percentuale"] = pd.Categorical(summary_df_new["Percentuale"], categories=thr_order, ordered=True)
summary_df_new = summary_df_new.sort_values(["Scenario", "Percentuale"]).reset_index(drop=True)

plot_frontiers_grid_by_fixedpct(summary_df_new, FIXED_PCTS, cols=3)
