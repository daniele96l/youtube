import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Lista ETF: (non-hedged, hedged)
etf_pairs = [
    ('CSPX.L', 'IUSE.MI'), #SP500 in dollar, sp500 in euro hedgato in dollari
    ('EQQQ.L', 'HNDX.DE'), #stessa cosa per il nasdaq
]

start_date = '2020-01-01'
end_date = '2025-07-10'
rolling_windows_years = [2,5]

# Funzione per calcolare CAGR
def calculate_cagr(start_price, end_price, years):
    return (end_price / start_price) ** (1 / years) - 1

# Per bar chart finale
hedge_costs = {}

for non_hedged, hedged in etf_pairs:
    # Download e preparazione dati
    tickers = [non_hedged, hedged]
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    normalized = data / data.iloc[0] * 100

    # --- Plot cumulativo normalizzato ---
    plt.figure(figsize=(12, 5))
    plt.plot(normalized[non_hedged], label=f'{non_hedged} (non-hedged)', color='blue')
    plt.plot(normalized[hedged], label=f'{hedged} (hedged)', color='green')
    plt.title(f'Performance Normalizzata: {non_hedged} vs {hedged}')
    plt.xlabel('Data')
    plt.ylabel('Valore Normalizzato (100 = inizio)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- CAGR totale ---
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    cagr_non_hedged = calculate_cagr(data[non_hedged].iloc[0], data[non_hedged].iloc[-1], years)
    cagr_hedged = calculate_cagr(data[hedged].iloc[0], data[hedged].iloc[-1], years)
    hedge_cost = (cagr_non_hedged - cagr_hedged) * 100
    hedge_costs[f'{non_hedged} vs {hedged}'] = hedge_cost

    print(f"\n{non_hedged} vs {hedged}")
    print(f"Periodo: {data.index[0].date()} → {data.index[-1].date()} ({years:.2f} anni)")
    print(f"CAGR {non_hedged}: {cagr_non_hedged*100:.2f}%")
    print(f"CAGR {hedged}:     {cagr_hedged*100:.2f}%")
    print(f"Stima costo medio annuo copertura: {hedge_cost:.2f}%")

    # --- Rolling CAGR ---
    for window_years in rolling_windows_years:
        window_days = int(window_years * 365.25)
        rolling_dates = data.index[data.index >= data.index[0] + pd.Timedelta(days=window_days)]

        rolling_hedge_costs = []
        rolling_labels = []

        for end_date_rolling in rolling_dates:
            start_date_rolling = end_date_rolling - pd.Timedelta(days=window_days)
            mask = (data.index >= start_date_rolling) & (data.index <= end_date_rolling)
            window_data = data.loc[mask]

            if len(window_data) < 2:
                continue

            window_cagr_non = calculate_cagr(window_data[non_hedged].iloc[0], window_data[non_hedged].iloc[-1], window_years)
            window_cagr_hedged = calculate_cagr(window_data[hedged].iloc[0], window_data[hedged].iloc[-1], window_years)
            rolling_hedge_costs.append((window_cagr_non - window_cagr_hedged) * 100)
            rolling_labels.append(end_date_rolling)

        # Plot rolling
        plt.figure(figsize=(12, 4))
        plt.plot(rolling_labels, rolling_hedge_costs, label=f'Rolling {window_years}Y Hedge Cost')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'{non_hedged} vs {hedged} - Rolling {window_years}Y Hedge Cost')
        plt.ylabel('Costo copertura (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Bar chart finale ---
plt.figure(figsize=(10, 5))
plt.bar(hedge_costs.keys(), hedge_costs.values(), color='salmon')
plt.ylabel('Costo Medio Annuo Copertura (%)')
plt.title('Confronto Costo Copertura ETF Hedged vs Non-Hedged')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()
