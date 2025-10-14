import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Scarica i dati per entrambi gli ETF
ticker1 = "XS2D.L"  # ETF con leva 2x
ticker2 = "SPY"   # S&P 500 (indice di riferimento)

# Scarica tutti i dati disponibili (max 5 anni)
data1 = yf.download(ticker1, period="max", progress=False, auto_adjust=True)
data2 = yf.download(ticker2, period="max", progress=False, auto_adjust=True)

# Handle multi-level columns when auto_adjust=True
if isinstance(data1.columns, pd.MultiIndex):
    price1 = data1['Close', ticker1]
    price2 = data2['Close', ticker2]
else:
    if 'Adj Close' in data1.columns:
        price1 = data1['Adj Close']
        price2 = data2['Adj Close']
    else:
        price1 = data1['Close']
        price2 = data2['Close']

print(f"Dati scaricati:")
print(f"{ticker1}: {len(price1)} giorni")
print(f"{ticker2}: {len(price2)} giorni")

# 2. Calcola i ritorni giornalieri
returns1 = price1.pct_change().dropna()
returns2 = price2.pct_change().dropna()

# Allinea le date (prendi solo le date comuni)
common_dates = returns1.index.intersection(returns2.index)
returns1_aligned = returns1.loc[common_dates]
returns2_aligned = returns2.loc[common_dates]

print(f"\nDate comuni: {len(common_dates)} giorni")

# 3. Calcola i ritorni 2x dell'indice
returns2_2x = returns2_aligned * 2

# 4. Create comprehensive analysis plots
fig = plt.figure(figsize=(20, 12))

# Create subplot layout: 2x3 grid
ax1 = plt.subplot(2, 3, (1, 2))  # Line chart spans 2 columns
ax2 = plt.subplot(2, 3, 3)       # CAGR bar chart
ax3 = plt.subplot(2, 3, 4)       # Risk metrics bar chart
ax4 = plt.subplot(2, 3, (5, 6))  # Drawdown chart spans 2 columns

# Plot 1: Prezzi normalizzati
price1_common = price1.loc[price1.index.isin(common_dates)]
price2_common = price2.loc[price2.index.isin(common_dates)]

# Calculate theoretical 2x leveraged S&P 500
price2_2x_theoretical = pd.Series(index=price2_common.index, dtype=float)
price2_2x_theoretical.iloc[0] = price2_common.iloc[0]
for i in range(1, len(price2_common)):
    price2_2x_theoretical.iloc[i] = price2_2x_theoretical.iloc[i-1] * (1 + returns2_2x.iloc[i-1])

# Plot 1: Line chart - Normalized prices
ax1.plot(price2_common.index, price2_common / price2_common.iloc[0] * 100, 
         label=f'{ticker2} (Normal S&P 500)', linewidth=2)
ax1.plot(price2_common.index, price2_2x_theoretical / price2_2x_theoretical.iloc[0] * 100, 
         label=f'{ticker2} x2 Theoretical', linewidth=2, linestyle='--')
ax1.plot(price1_common.index, price1_common / price1_common.iloc[0] * 100, 
         label=f'{ticker1} (Real Leveraged ETF)', linewidth=2)
ax1.set_title('Normalized Price Evolution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Normalized Value (Base 100)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart - CAGR comparison
# Calculate CAGR
years = (price2_common.index[-1] - price2_common.index[0]).days / 365.25

spy_cagr = (price2_common.iloc[-1] / price2_common.iloc[0]) ** (1/years) - 1
etf_cagr = (price1_common.iloc[-1] / price1_common.iloc[0]) ** (1/years) - 1
theoretical_cagr = (price2_2x_theoretical.iloc[-1] / price2_2x_theoretical.iloc[0]) ** (1/years) - 1

categories = [f'{ticker2}\n(Normal)', f'{ticker2}\n(2x Theoretical)', f'{ticker1}\n(Real ETF)']
cagr_values = [spy_cagr, theoretical_cagr, etf_cagr]
colors = ['blue', 'orange', 'green']

bars = ax2.bar(categories, cagr_values, color=colors, alpha=0.7)
ax2.set_title('CAGR Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Compound Annual Growth Rate', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, cagr_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Format y-axis as percentage
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# Plot 3: Risk metrics bar chart (Volatility, Sharpe, Sortino)
# Calculate risk metrics
risk_free_rate = 0.02  # Assume 2% annual risk-free rate
annual_risk_free = risk_free_rate / 252  # Daily risk-free rate

# Volatility (annualized)
spy_vol = returns2_aligned.std() * np.sqrt(252)
etf_vol = returns1_aligned.std() * np.sqrt(252)
theoretical_vol = (returns2_aligned * 2).std() * np.sqrt(252)

# Sharpe Ratio
spy_sharpe = (spy_cagr - risk_free_rate) / spy_vol
etf_sharpe = (etf_cagr - risk_free_rate) / etf_vol
theoretical_sharpe = (theoretical_cagr - risk_free_rate) / theoretical_vol

# Sortino Ratio (using downside deviation)
spy_downside = returns2_aligned[returns2_aligned < 0].std() * np.sqrt(252)
etf_downside = returns1_aligned[returns1_aligned < 0].std() * np.sqrt(252)
theoretical_downside = (returns2_aligned * 2)[(returns2_aligned * 2) < 0].std() * np.sqrt(252)

spy_sortino = (spy_cagr - risk_free_rate) / spy_downside
etf_sortino = (etf_cagr - risk_free_rate) / etf_downside
theoretical_sortino = (theoretical_cagr - risk_free_rate) / theoretical_downside

# Create grouped bar chart
x = np.arange(len(categories))
width = 0.25

vol_values = [spy_vol, theoretical_vol, etf_vol]
sharpe_values = [spy_sharpe, theoretical_sharpe, etf_sharpe]
sortino_values = [spy_sortino, theoretical_sortino, etf_sortino]

bars1 = ax3.bar(x - width, vol_values, width, label='Volatility', alpha=0.7, color='red')
bars2 = ax3.bar(x, sharpe_values, width, label='Sharpe Ratio', alpha=0.7, color='blue')
bars3 = ax3.bar(x + width, sortino_values, width, label='Sortino Ratio', alpha=0.7, color='green')

ax3.set_title('Risk Metrics Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('Value', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels([f'{ticker2}\n(Normal)', f'{ticker2}\n(2x Theoretical)', f'{ticker1}\n(Real ETF)'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Drawdown evolution
# Calculate drawdowns
spy_cumulative = (1 + returns2_aligned).cumprod()
etf_cumulative = (1 + returns1_aligned).cumprod()
theoretical_cumulative = (1 + returns2_aligned * 2).cumprod()

spy_drawdown = (spy_cumulative / spy_cumulative.cummax() - 1) * 100
etf_drawdown = (etf_cumulative / etf_cumulative.cummax() - 1) * 100
theoretical_drawdown = (theoretical_cumulative / theoretical_cumulative.cummax() - 1) * 100

ax4.fill_between(common_dates, spy_drawdown, 0, alpha=0.3, color='blue', label=f'{ticker2} Drawdown')
ax4.fill_between(common_dates, etf_drawdown, 0, alpha=0.3, color='green', label=f'{ticker1} Drawdown')
ax4.fill_between(common_dates, theoretical_drawdown, 0, alpha=0.3, color='orange', label=f'{ticker2} 2x Theoretical Drawdown')

ax4.set_title('Drawdown Evolution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12)
ax4.set_ylabel('Drawdown (%)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
import os
output_path = os.path.join(os.path.dirname(__file__), 'leverage_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# 5. Statistiche di confronto
print("\n" + "="*60)
print("ANALISI DEI RITORNI")
print("="*60)
print(f"\nRitorno medio giornaliero {ticker2}: {returns2_aligned.mean():.4%}")
print(f"Ritorno medio giornaliero {ticker1}: {returns1_aligned.mean():.4%}")
print(f"Ritorno medio giornaliero {ticker2} x2 (teorico): {returns2_2x.mean():.4%}")

print(f"\nVolatilità (std dev) {ticker2}: {returns2_aligned.std():.4%}")
print(f"Volatilità (std dev) {ticker1}: {returns1_aligned.std():.4%}")
print(f"Volatilità (std dev) {ticker2} x2 (teorico): {returns2_2x.std():.4%}")

# Correlazione
correlation = np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]
print(f"\nCorrelazione tra {ticker1} e {ticker2}: {correlation:.4f}")

# Differenza media
difference = returns1_aligned - returns2_2x
mean_diff = difference.mean()
print(f"\nDifferenza media (Effettivo - Teorico): {mean_diff:.4%}")
print(f"Differenza std dev: {difference.std():.4%}")

# R-squared del modello 2x (calcolo manuale)
ss_res = np.sum((returns1_aligned - returns2_2x) ** 2)
ss_tot = np.sum((returns1_aligned - returns1_aligned.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nR² (quanto bene il modello 2x spiega i ritorni effettivi): {r2:.4f}")

# Calcolo del rapporto di leva effettivo
slope, intercept = np.polyfit(returns2_aligned, returns1_aligned, 1)
print(f"\nLeva effettiva (slope della regressione): {slope:.4f}x")
print(f"Intercetta della regressione: {intercept:.6f}")
print(f"Target leva: 2.00x")
print(f"Differenza dalla leva target: {(slope - 2.0):.4f}x ({(slope - 2.0)/2.0 * 100:.2f}%)")

print("\n" + "="*60)