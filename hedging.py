import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ETF: CSPX (USD), IUSE (EUR Hedged)
tickers = ['CSPX.L', 'IUSE.MI']
start_date = '2020-01-01'
end_date = '2025-07-10'

# Scarica dati Close
data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()

# Normalizza per grafico
normalized = data / data.iloc[0] * 100

# Calcolo durate in anni
days = (data.index[-1] - data.index[0]).days
years = days / 365.25

# Calcola CAGR
start_prices = data.iloc[0]
end_prices = data.iloc[-1]

cagr_cspx = ((end_prices['CSPX.L'] / start_prices['CSPX.L']) ** (1 / years)) - 1
cagr_iuse = ((end_prices['IUSE.MI'] / start_prices['IUSE.MI']) ** (1 / years)) - 1
hedge_cost_estimate = (cagr_cspx - cagr_iuse) * 100  # in percentuale

# Plot
plt.figure(figsize=(12, 6))
plt.plot(normalized['CSPX.L'], label='CSPX (USD, non-hedged)', color='blue')
plt.plot(normalized['IUSE.MI'], label='IUSE (EUR Hedged)', color='green')
plt.title('Performance Normalizzata - CSPX vs IUSE (Close)')
plt.xlabel('Data')
plt.ylabel('Valore Normalizzato (100 = inizio)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output dei risultati
print(f"Periodo: {data.index[0].date()} → {data.index[-1].date()} ({years:.2f} anni)")
print(f"CAGR CSPX (non-hedged): {cagr_cspx * 100:.2f}%")
print(f"CAGR IUSE (hedged):     {cagr_iuse * 100:.2f}%")
print(f"Stima costo medio annuo copertura: {hedge_cost_estimate:.2f}%")
