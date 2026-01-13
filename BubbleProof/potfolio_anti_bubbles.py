import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

# Function to blend colors based on allocation percentages
def blend_colors(acwi_pct, bond_pct, gold_pct):
    """
    Blend colors: Gold=Yellow, ACWI=Red, Bond=Blue
    Returns RGB tuple normalized to [0, 1]
    """
    # Define base colors in RGB (0-1 range)
    gold_rgb = np.array([1.0, 1.0, 0.0])  # Yellow
    acwi_rgb = np.array([1.0, 0.0, 0.0])  # Red
    bond_rgb = np.array([0.0, 0.0, 1.0])  # Blue
    
    # Blend colors weighted by allocation percentages
    blended = (gold_pct * gold_rgb + acwi_pct * acwi_rgb + bond_pct * bond_rgb)
    
    # Ensure values are in [0, 1] range
    blended = np.clip(blended, 0, 1)
    
    return tuple(blended)

# Load data
acwi = pd.read_csv('acwi.csv', parse_dates=['Date'], dayfirst=False)
bond = pd.read_csv('bond.csv', parse_dates=['Date'], dayfirst=False)
gold = pd.read_csv('gold.csv', parse_dates=['Date'], dayfirst=False)

# Rename columns for easier access
acwi.columns = ['Date', 'ACWI']
bond.columns = ['Date', 'Bond']
gold.columns = ['Date', 'Gold']

# Merge all dataframes
df = acwi.merge(bond, on='Date', how='outer')
df = df.merge(gold, on='Date', how='outer')
df = df.sort_values('Date').reset_index(drop=True)

# Forward fill missing values
df = df.fillna(method='ffill').dropna()

# Calculate returns
df['ACWI_ret'] = df['ACWI'].pct_change()
df['Bond_ret'] = df['Bond'].pct_change()
df['Gold_ret'] = df['Gold'].pct_change()

# Generate all portfolio combinations (25% steps, must sum to 100%, 0% is allowed)
allocations = []
for acwi_pct in range(0, 101, 25):
    for bond_pct in range(0, 101, 25):
        gold_pct = 100 - acwi_pct - bond_pct
        if gold_pct >= 0:  # Allow 0% allocations
            allocations.append({
                'ACWI': acwi_pct / 100,
                'Bond': bond_pct / 100,
                'Gold': gold_pct / 100
            })

print(f"Generated {len(allocations)} portfolio allocations")

# Calculate portfolio returns for each allocation
portfolio_returns = []
for i, alloc in enumerate(allocations):
    df[f'Portfolio_{i}'] = (
        df['ACWI_ret'] * alloc['ACWI'] +
        df['Bond_ret'] * alloc['Bond'] +
        df['Gold_ret'] * alloc['Gold']
    )
    portfolio_returns.append({
        'Portfolio': i,
        'ACWI': alloc['ACWI'],
        'Bond': alloc['Bond'],
        'Gold': alloc['Gold']
    })

portfolio_df = pd.DataFrame(portfolio_returns)

# Calculate cumulative performance for each portfolio
for i in range(len(allocations)):
    df[f'Portfolio_{i}_cum'] = (1 + df[f'Portfolio_{i}']).cumprod() * 10000

# Define crisis periods (2 years from start date)
crises = {
    '1990': {'start': '1989-01-01', 'end': '1991-01-01', 'name': 'Savings & Loan Crisis / Early 1990s Recession'},
    '1997': {'start': '1997-07-01', 'end': '1999-07-01', 'name': 'Asian Financial Crisis'},
    '2000': {'start': '2000-01-01', 'end': '2002-01-01', 'name': 'Dot-com Bubble (2000-2003)'},
    '2008': {'start': '2008-01-01', 'end': '2010-01-01', 'name': 'Financial Crisis (2008-2011)'},
    '2010': {'start': '2010-01-01', 'end': '2012-01-01', 'name': 'European Sovereign Debt Crisis'},
    '2020': {'start': '2020-02-01', 'end': '2022-02-01', 'name': 'COVID-19 Market Crash'},
    '2022': {'start': '2022-01-01', 'end': '2024-01-01', 'name': 'Inflation-Driven Global Bear Market'}
}

# Analyze performance during and after crises
crisis_results = []

for crisis_key, crisis_info in crises.items():
    start_date = pd.to_datetime(crisis_info['start'])
    end_date = pd.to_datetime(crisis_info['end'])
    
    # Filter data for crisis period
    crisis_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    if len(crisis_data) == 0:
        continue
    
    # Calculate performance metrics for each portfolio
    for i in range(len(allocations)):
        if f'Portfolio_{i}_cum' in crisis_data.columns:
            start_val = crisis_data[f'Portfolio_{i}_cum'].iloc[0]
            end_val = crisis_data[f'Portfolio_{i}_cum'].iloc[-1]
            total_return = (end_val / start_val - 1) * 100
            
            # Calculate max drawdown
            cummax = crisis_data[f'Portfolio_{i}_cum'].cummax()
            drawdown = ((crisis_data[f'Portfolio_{i}_cum'] - cummax) / cummax * 100).min()
            
            # Calculate CAGR
            years = (end_date - start_date).days / 365.25
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else 0
            
            # Calculate volatility (annualized)
            portfolio_returns = crisis_data[f'Portfolio_{i}'].dropna()
            if len(portfolio_returns) > 1:
                # Assume monthly data (12 periods per year)
                periods_per_year = 12
                volatility = portfolio_returns.std() * np.sqrt(periods_per_year) * 100
            else:
                volatility = 0
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            sharpe = (cagr / 100) / (volatility / 100) if volatility > 0 else 0
            
            crisis_results.append({
                'Crisis': crisis_key,
                'Portfolio': i,
                'ACWI': portfolio_df.iloc[i]['ACWI'],
                'Bond': portfolio_df.iloc[i]['Bond'],
                'Gold': portfolio_df.iloc[i]['Gold'],
                'Total_Return': total_return,
                'Max_Drawdown': drawdown,
                'CAGR': cagr,
                'Volatility': volatility,
                'Sharpe': sharpe
            })

crisis_results_df = pd.DataFrame(crisis_results)

# Calculate drawdown over time for each portfolio during each crisis
crisis_drawdowns = {}
for crisis_key, crisis_info in crises.items():
    start_date = pd.to_datetime(crisis_info['start'])
    end_date = pd.to_datetime(crisis_info['end'])
    crisis_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    if len(crisis_data) == 0:
        continue
    
    drawdown_data = {'Date': crisis_data['Date']}
    for i in range(len(allocations)):
        if f'Portfolio_{i}_cum' in crisis_data.columns:
            cummax = crisis_data[f'Portfolio_{i}_cum'].cummax()
            drawdown = ((crisis_data[f'Portfolio_{i}_cum'] - cummax) / cummax * 100)
            drawdown_data[f'Portfolio_{i}'] = drawdown.values
    
    crisis_drawdowns[crisis_key] = pd.DataFrame(drawdown_data)

# Create separate figures for each crisis
for crisis_key, crisis_info in crises.items():
    start_date = pd.to_datetime(crisis_info['start'])
    end_date = pd.to_datetime(crisis_info['end'])
    crisis_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    if len(crisis_data) == 0:
        continue
    
    crisis_results_subset = crisis_results_df[crisis_results_df['Crisis'] == crisis_key]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{crisis_info["name"]} Analysis', fontsize=16, fontweight='bold')
    
    # 1. Max Drawdown as line chart (over time)
    ax3 = axes[0, 0]
    if crisis_key in crisis_drawdowns:
        drawdown_df = crisis_drawdowns[crisis_key]
        # Plot all portfolios with blended colors based on allocation
        for i in range(len(allocations)):
            if f'Portfolio_{i}' in drawdown_df.columns:
                alloc = allocations[i]
                # Blend colors based on allocation percentages
                color = blend_colors(alloc['ACWI'], alloc['Bond'], alloc['Gold'])
                
                # Determine linestyle based on dominant asset
                max_alloc = max(alloc['ACWI'], alloc['Bond'], alloc['Gold'])
                if alloc['Gold'] == max_alloc:
                    linestyle = '-'
                elif alloc['ACWI'] == max_alloc:
                    linestyle = '--'
                else:  # Bond is dominant
                    linestyle = '-.'
                
                label = f'ACWI:{alloc["ACWI"]:.0%} Bond:{alloc["Bond"]:.0%} Gold:{alloc["Gold"]:.0%}'
                ax3.plot(
                    drawdown_df['Date'],
                    drawdown_df[f'Portfolio_{i}'],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.7,
                    linewidth=1.5
                )
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Drawdown Over Time\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax3.tick_params(axis='x', rotation=45)
    
    # 2. Portfolio value evolution over time
    ax4 = axes[0, 1]
    
    # Hide the empty subplot in row 1
    axes[0, 2].axis('off')
    # Plot all portfolios with blended colors based on allocation
    for i in range(len(allocations)):
        if f'Portfolio_{i}_cum' in crisis_data.columns:
            alloc = allocations[i]
            # Blend colors based on allocation percentages
            color = blend_colors(alloc['ACWI'], alloc['Bond'], alloc['Gold'])
            
            # Determine linestyle based on dominant asset
            max_alloc = max(alloc['ACWI'], alloc['Bond'], alloc['Gold'])
            if alloc['Gold'] == max_alloc:
                linestyle = '-'
            elif alloc['ACWI'] == max_alloc:
                linestyle = '--'
            else:  # Bond is dominant
                linestyle = '-.'
            
            normalized = crisis_data[f'Portfolio_{i}_cum'] / crisis_data[f'Portfolio_{i}_cum'].iloc[0] * 10000
            label = f'ACWI:{alloc["ACWI"]:.0%} Bond:{alloc["Bond"]:.0%} Gold:{alloc["Gold"]:.0%}'
            ax4.plot(
                crisis_data['Date'],
                normalized,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=0.7,
                linewidth=1.5
            )
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Portfolio Value (Normalized to 10,000)')
    ax4.set_title('Portfolio Value Evolution\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax4.tick_params(axis='x', rotation=45)
    
    # Prepare data for bar charts
    portfolio_labels = []
    cagr_values = []
    vol_values = []
    sharpe_values = []
    colors_bar = []
    
    for i in range(len(allocations)):
        portfolio_metrics = crisis_results_subset[crisis_results_subset['Portfolio'] == i]
        if len(portfolio_metrics) > 0:
            alloc = allocations[i]
            portfolio_labels.append(f'A:{alloc["ACWI"]:.0%} B:{alloc["Bond"]:.0%} G:{alloc["Gold"]:.0%}')
            cagr_values.append(portfolio_metrics['CAGR'].iloc[0])
            vol_values.append(portfolio_metrics['Volatility'].iloc[0])
            sharpe_values.append(portfolio_metrics['Sharpe'].iloc[0])
            
            # Blend colors based on allocation percentages
            colors_bar.append(blend_colors(alloc['ACWI'], alloc['Bond'], alloc['Gold']))
    
    # Sort data for each metric
    # CAGR: sort descending (highest first)
    sorted_cagr = sorted(zip(cagr_values, portfolio_labels, colors_bar), key=lambda x: x[0], reverse=True)
    cagr_sorted, labels_cagr, colors_cagr = zip(*sorted_cagr)
    
    # Volatility: sort ascending (lowest first)
    sorted_vol = sorted(zip(vol_values, portfolio_labels, colors_bar), key=lambda x: x[0])
    vol_sorted, labels_vol, colors_vol = zip(*sorted_vol)
    
    # Sharpe: sort descending (highest first)
    sorted_sharpe = sorted(zip(sharpe_values, portfolio_labels, colors_bar), key=lambda x: x[0], reverse=True)
    sharpe_sorted, labels_sharpe, colors_sharpe = zip(*sorted_sharpe)
    
    x = np.arange(len(portfolio_labels))
    
    # 3. CAGR bar chart
    ax5 = axes[1, 0]
    ax5.bar(x, cagr_sorted, color=colors_cagr, alpha=0.7)
    ax5.set_xlabel('Portfolio Allocation')
    ax5.set_ylabel('CAGR (%)')
    ax5.set_title('CAGR by Portfolio\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels_cagr, rotation=45, ha='right', fontsize=7)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 4. Volatility bar chart
    ax6 = axes[1, 1]
    ax6.bar(x, vol_sorted, color=colors_vol, alpha=0.7)
    ax6.set_xlabel('Portfolio Allocation')
    ax6.set_ylabel('Volatility (%)')
    ax6.set_title('Volatility by Portfolio\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels_vol, rotation=45, ha='right', fontsize=7)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 5. Sharpe Ratio bar chart
    ax7 = axes[1, 2]
    ax7.bar(x, sharpe_sorted, color=colors_sharpe, alpha=0.7)
    ax7.set_xlabel('Portfolio Allocation')
    ax7.set_ylabel('Sharpe Ratio')
    ax7.set_title('Sharpe Ratio by Portfolio\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(labels_sharpe, rotation=45, ha='right', fontsize=7)
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'portfolio_anti_bubbles_{crisis_key}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {crisis_info['name']} analysis to '{filename}'")
    plt.close()

print("Analysis complete! All plots saved.")

# Create summary statistics
print("\n=== Summary Statistics ===")
for crisis_key in crises.keys():
    crisis_data = crisis_results_df[crisis_results_df['Crisis'] == crisis_key]
    if len(crisis_data) > 0 and not crisis_data['Total_Return'].isna().all():
        valid_data = crisis_data.dropna(subset=['Total_Return'])
        if len(valid_data) > 0:
            print(f"\n{crises[crisis_key]['name']}:")
            print(f"  Best Return: {valid_data['Total_Return'].max():.2f}%")
            print(f"  Worst Return: {valid_data['Total_Return'].min():.2f}%")
            print(f"  Average Return: {valid_data['Total_Return'].mean():.2f}%")
            best_idx = valid_data['Total_Return'].idxmax()
            print(f"  Best Portfolio (by return): ACWI={valid_data.loc[best_idx, 'ACWI']:.0%}, "
                  f"Bond={valid_data.loc[best_idx, 'Bond']:.0%}, "
                  f"Gold={valid_data.loc[best_idx, 'Gold']:.0%}")

# Create final summary plot with averages across all crises
print("\nCreating final summary plot with averages across all crises...")

# Calculate average metrics for each portfolio across all crises
portfolio_summary = []
for i in range(len(allocations)):
    portfolio_crises = crisis_results_df[crisis_results_df['Portfolio'] == i].dropna(subset=['CAGR', 'Volatility', 'Sharpe'])
    if len(portfolio_crises) > 0:
        portfolio_summary.append({
            'Portfolio': i,
            'ACWI': allocations[i]['ACWI'],
            'Bond': allocations[i]['Bond'],
            'Gold': allocations[i]['Gold'],
            'Avg_CAGR': portfolio_crises['CAGR'].mean(),
            'Avg_Volatility': portfolio_crises['Volatility'].mean(),
            'Avg_Sharpe': portfolio_crises['Sharpe'].mean()
        })

portfolio_summary_df = pd.DataFrame(portfolio_summary)

# Create summary plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Average Performance Across All Crises', fontsize=16, fontweight='bold')

# Prepare data
portfolio_labels = []
cagr_avg = []
vol_avg = []
sharpe_avg = []
colors_bar = []

for _, row in portfolio_summary_df.iterrows():
    portfolio_idx = int(row['Portfolio'])
    alloc = allocations[portfolio_idx]
    portfolio_labels.append(f'A:{alloc["ACWI"]:.0%} B:{alloc["Bond"]:.0%} G:{alloc["Gold"]:.0%}')
    cagr_avg.append(row['Avg_CAGR'])
    vol_avg.append(row['Avg_Volatility'])
    sharpe_avg.append(row['Avg_Sharpe'])
    colors_bar.append(blend_colors(alloc['ACWI'], alloc['Bond'], alloc['Gold']))

# Sort data for each metric
# CAGR: sort descending (highest first)
sorted_cagr = sorted(zip(cagr_avg, portfolio_labels, colors_bar), key=lambda x: x[0], reverse=True)
cagr_sorted, labels_cagr, colors_cagr = zip(*sorted_cagr)

# Volatility: sort ascending (lowest first)
sorted_vol = sorted(zip(vol_avg, portfolio_labels, colors_bar), key=lambda x: x[0])
vol_sorted, labels_vol, colors_vol = zip(*sorted_vol)

# Sharpe: sort descending (highest first)
sorted_sharpe = sorted(zip(sharpe_avg, portfolio_labels, colors_bar), key=lambda x: x[0], reverse=True)
sharpe_sorted, labels_sharpe, colors_sharpe = zip(*sorted_sharpe)

x = np.arange(len(portfolio_labels))

# CAGR bar chart
ax1 = axes[0]
ax1.bar(x, cagr_sorted, color=colors_cagr, alpha=0.7)
ax1.set_xlabel('Portfolio Allocation')
ax1.set_ylabel('Average CAGR (%)')
ax1.set_title('Average CAGR Across All Crises\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_cagr, rotation=45, ha='right', fontsize=7)
ax1.grid(True, alpha=0.3, axis='y')

# Volatility bar chart
ax2 = axes[1]
ax2.bar(x, vol_sorted, color=colors_vol, alpha=0.7)
ax2.set_xlabel('Portfolio Allocation')
ax2.set_ylabel('Average Volatility (%)')
ax2.set_title('Average Volatility Across All Crises\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels_vol, rotation=45, ha='right', fontsize=7)
ax2.grid(True, alpha=0.3, axis='y')

# Sharpe Ratio bar chart
ax3 = axes[2]
ax3.bar(x, sharpe_sorted, color=colors_sharpe, alpha=0.7)
ax3.set_xlabel('Portfolio Allocation')
ax3.set_ylabel('Average Sharpe Ratio')
ax3.set_title('Average Sharpe Ratio Across All Crises\n(Gold=Yellow, ACWI=Red, Bond=Blue)')
ax3.set_xticks(x)
ax3.set_xticklabels(labels_sharpe, rotation=45, ha='right', fontsize=7)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
filename = 'portfolio_anti_bubbles_summary.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved summary plot to '{filename}'")
plt.close()

