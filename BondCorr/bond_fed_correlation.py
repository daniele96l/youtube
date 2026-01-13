import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Create output folders for each bond duration
output_folders = {
    '1-3_year': 'output_1_3_year',
    '3-7_year': 'output_3_7_year',
    '20_plus_year': 'output_20_plus_year'
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Load data
print("\nLoading data...")
fed_rate = pd.read_csv('data/fed_funds_rate_monthly_1954_2025_cleaned.csv')
bond_1_3 = pd.read_csv('data/ICE US Treasury 1-3 Year Bond index.csv')
bond_3_7 = pd.read_csv('data/ICE US Treasury 3-7 Year Bond index.csv')
bond_20_plus = pd.read_csv('data/ICE US Treasury 20+ Year Bond index.csv')

# Parse dates
fed_rate['Date'] = pd.to_datetime(fed_rate['Date'], format='%m/%Y')
bond_1_3['Date'] = pd.to_datetime(bond_1_3['Date'], format='%m/%Y')
bond_3_7['Date'] = pd.to_datetime(bond_3_7['Date'], format='%m/%Y')
bond_20_plus['Date'] = pd.to_datetime(bond_20_plus['Date'], format='%m/%Y')

# Set Date as index
fed_rate.set_index('Date', inplace=True)
bond_1_3.set_index('Date', inplace=True)
bond_3_7.set_index('Date', inplace=True)
bond_20_plus.set_index('Date', inplace=True)

# Rename columns for clarity
bond_1_3.columns = ['Bond_Price']
bond_3_7.columns = ['Bond_Price']
bond_20_plus.columns = ['Bond_Price']
fed_rate.columns = ['Fed_Rate']

# Merge data for each bond separately
print("Merging data...")
data_1_3 = fed_rate.join(bond_1_3, how='inner')
data_3_7 = fed_rate.join(bond_3_7, how='inner')
data_20_plus = fed_rate.join(bond_20_plus, how='inner')

# Function to analyze and plot for each bond
def analyze_bond(data, bond_name, output_folder, bond_label):
    # Create a copy to avoid SettingWithCopyWarning
    data = data.copy()
    # Calculate monthly returns (percentage change) for bonds
    data['Bond_Return'] = data['Bond_Price'].pct_change() * 100
    
    # Remove first row (NaN from pct_change)
    data = data.dropna()
    
    print(f"\n=== {bond_label} ANALYSIS ===")
    print(f"Data period: {data.index.min()} to {data.index.max()}")
    print(f"Total months: {len(data)}")
    
    # Calculate correlation with Fed Rate (level, not change)
    corr = data['Bond_Return'].corr(data['Fed_Rate'])
    print(f"\nCorrelation between {bond_label} Bond Returns and Fed Rate: {corr:.4f}")
    
    # Statistics
    print(f"\nFed Rate Statistics:")
    print(data['Fed_Rate'].describe())
    print(f"\nBond Return Statistics:")
    print(data['Bond_Return'].describe())
    
    # Identify "crash" periods (large negative returns)
    crash_threshold = -2.0
    data['Bond_Crash'] = data['Bond_Return'] < crash_threshold
    
    crash_count = data['Bond_Crash'].sum()
    print(f"\nCrash periods (>-2%): {crash_count} months")
    
    # Average Fed rate during crashes vs normal
    avg_fed_during_crash = data[data['Bond_Crash']]['Fed_Rate'].mean() if crash_count > 0 else np.nan
    avg_fed_normal = data[~data['Bond_Crash']]['Fed_Rate'].mean()
    print(f"Average Fed Rate during crashes: {avg_fed_during_crash:.2f}%")
    print(f"Average Fed Rate during normal periods: {avg_fed_normal:.2f}%")
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time series of Fed Rate and Bond Price
    ax1 = plt.subplot(3, 2, 1)
    ax1_twin = ax1.twinx()
    ax1.plot(data.index, data['Fed_Rate'], 'b-', label='Fed Funds Rate', linewidth=2)
    ax1_twin.plot(data.index, data['Bond_Price'], 'r-', label=f'{bond_label} Bond Price', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Fed Funds Rate (%)', color='b')
    ax1_twin.set_ylabel('Bond Price (Index)', color='r')
    ax1.set_title(f'Fed Funds Rate vs {bond_label} Bond Price Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Scatter: Fed Rate vs Bond Returns
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(data['Fed_Rate'], data['Bond_Return'], alpha=0.6, s=40, color='steelblue')
    # Add trend line
    z = np.polyfit(data['Fed_Rate'], data['Bond_Return'], 1)
    p = np.poly1d(z)
    ax2.plot(data['Fed_Rate'], p(data['Fed_Rate']), "r--", alpha=0.8, linewidth=2, 
             label=f'Trend (corr={corr:.3f})')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Fed Funds Rate (%)')
    ax2.set_ylabel('Bond Return (%)')
    ax2.set_title(f'Fed Rate vs {bond_label} Bond Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation heatmap
    ax3 = plt.subplot(3, 2, 3)
    corr_matrix = data[['Fed_Rate', 'Bond_Return']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, ax=ax3, cbar_kws={'label': 'Correlation'})
    ax3.set_title('Correlation Matrix')
    
    # 4. Fed Rate distribution during crashes vs normal
    ax4 = plt.subplot(3, 2, 4)
    if crash_count > 0:
        crash_data = data[data['Bond_Crash']]['Fed_Rate']
        normal_data = data[~data['Bond_Crash']]['Fed_Rate']
        ax4.hist(normal_data, bins=30, alpha=0.6, label='Normal Periods', color='green', density=True)
        ax4.hist(crash_data, bins=30, alpha=0.6, label='Crash Periods (>-2%)', color='red', density=True)
        ax4.set_xlabel('Fed Funds Rate (%)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Fed Rate Distribution: Crashes vs Normal ({bond_label})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No crash periods detected', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title(f'Fed Rate Distribution: Crashes vs Normal ({bond_label})')
    
    # 5. Rolling correlation (12-month window)
    ax5 = plt.subplot(3, 2, 5)
    window = 12
    rolling_corr = data['Bond_Return'].rolling(window=window).corr(data['Fed_Rate'])
    ax5.plot(data.index, rolling_corr, label=f'{bond_label}', linewidth=2, color='steelblue')
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Rolling Correlation (12-month)')
    ax5.set_title(f'Rolling Correlation: {bond_label} Bond Returns vs Fed Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Bond returns over time with crash periods highlighted
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(data.index, data['Bond_Return'], 'b-', alpha=0.6, linewidth=1, label=f'{bond_label} Return')
    if crash_count > 0:
        crash_dates = data[data['Bond_Crash']].index
        crash_returns = data[data['Bond_Crash']]['Bond_Return']
        ax6.scatter(crash_dates, crash_returns, color='red', s=50, zorder=5, label='Crash Periods')
    ax6.axhline(y=crash_threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, 
                label=f'Crash Threshold ({crash_threshold}%)')
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Bond Return (%)')
    ax6.set_title(f'{bond_label} Bond Returns with Crash Periods Highlighted')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{bond_name}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis saved to '{output_path}'")
    plt.close()
    
    return corr, data

# Function to remove outliers using IQR method
def remove_outliers_iqr(data, columns):
    """Remove outliers using Interquartile Range method"""
    data_clean = data.copy()
    for col in columns:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_clean = data_clean[(data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)]
    return data_clean

# Function for in-depth analysis: Fed rate deviation from 12-month average
def analyze_fed_deviation(data, bond_name, output_folder, bond_label):
    # Create a copy
    data = data.copy()
    
    # Calculate monthly returns
    data['Bond_Return'] = data['Bond_Price'].pct_change() * 100
    
    # Remove NaN rows (first row from pct_change)
    data = data.dropna()
    
    # Calculate 12-month rolling average of Fed rate
    window = 12
    data['Fed_Rate_12M_Avg'] = data['Fed_Rate'].rolling(window=window).mean()
    
    # Calculate deviation from 12-month average
    data['Fed_Rate_Deviation'] = data['Fed_Rate'] - data['Fed_Rate_12M_Avg']
    
    # Smooth bond returns using 12-month rolling average (yearly smoothing)
    data['Bond_Return_Smoothed'] = data['Bond_Return'].rolling(window=window).mean()
    
    # Remove NaN rows (first 12 months for rolling averages)
    data = data.dropna()
    
    # Remove outliers from monthly data
    data_clean = remove_outliers_iqr(data, ['Bond_Return', 'Fed_Rate_Deviation'])
    print(f"Removed {len(data) - len(data_clean)} outliers ({100*(len(data) - len(data_clean))/len(data):.1f}%)")
    
    # Resample to yearly data (end of year) - use smoothed values, then remove outliers
    data_yearly = data[['Fed_Rate_Deviation', 'Bond_Return_Smoothed']].resample('Y').last()
    data_yearly_clean = remove_outliers_iqr(data_yearly, ['Bond_Return_Smoothed', 'Fed_Rate_Deviation'])
    print(f"Removed {len(data_yearly) - len(data_yearly_clean)} yearly outliers ({100*(len(data_yearly) - len(data_yearly_clean))/len(data_yearly):.1f}%)")
    
    print(f"\n=== {bond_label} FED DEVIATION ANALYSIS ===")
    print(f"Data period: {data_clean.index.min()} to {data_clean.index.max()}")
    print(f"Total months: {len(data_clean)} (after outlier removal)")
    print(f"Total years: {len(data_yearly_clean)} (after outlier removal)")
    
    # Calculate correlations (using cleaned data for better trend analysis)
    corr_deviation = data_clean['Bond_Return'].corr(data_clean['Fed_Rate_Deviation'])
    corr_deviation_smoothed = data_yearly_clean['Bond_Return_Smoothed'].corr(data_yearly_clean['Fed_Rate_Deviation'])
    
    print(f"\nCorrelation between Bond Returns and Fed Rate Deviation (outliers removed): {corr_deviation:.4f}")
    print(f"Correlation between Smoothed Bond Returns and Fed Rate Deviation (outliers removed): {corr_deviation_smoothed:.4f}")
    
    # Analyze when Fed rate is above vs below average (using cleaned data)
    above_avg = data_clean[data_clean['Fed_Rate_Deviation'] > 0]
    below_avg = data_clean[data_clean['Fed_Rate_Deviation'] < 0]
    
    print(f"\nWhen Fed Rate > 12M Average:")
    print(f"  Count: {len(above_avg)} months")
    print(f"  Average Bond Return: {above_avg['Bond_Return'].mean():.4f}%")
    print(f"  Std Bond Return: {above_avg['Bond_Return'].std():.4f}%")
    
    print(f"\nWhen Fed Rate < 12M Average:")
    print(f"  Count: {len(below_avg)} months")
    print(f"  Average Bond Return: {below_avg['Bond_Return'].mean():.4f}%")
    print(f"  Std Bond Return: {below_avg['Bond_Return'].std():.4f}%")
    
    # Create comprehensive visualization (3x2 layout, removed plot #6)
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Time series: Fed Rate, 12M Average, and Deviation (with trend)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data_clean.index, data_clean['Fed_Rate'], 'b-', label='Fed Funds Rate', linewidth=2, alpha=0.7)
    ax1.plot(data_clean.index, data_clean['Fed_Rate_12M_Avg'], 'r--', label='12-Month Average', linewidth=2)
    # Add trend line for Fed Rate
    fed_trend = data_clean['Fed_Rate'].rolling(window=24).mean()
    ax1.plot(data_clean.index, fed_trend, 'k:', linewidth=2, alpha=0.8, label='24-Month Trend')
    ax1.fill_between(data_clean.index, data_clean['Fed_Rate'], data_clean['Fed_Rate_12M_Avg'], 
                     where=(data_clean['Fed_Rate'] >= data_clean['Fed_Rate_12M_Avg']), 
                     alpha=0.3, color='red', label='Above Average')
    ax1.fill_between(data_clean.index, data_clean['Fed_Rate'], data_clean['Fed_Rate_12M_Avg'], 
                     where=(data_clean['Fed_Rate'] < data_clean['Fed_Rate_12M_Avg']), 
                     alpha=0.3, color='green', label='Below Average')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Fed Funds Rate (%)')
    ax1.set_title(f'Fed Rate vs 12-Month Average ({bond_label})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter: Fed Rate Deviation vs Bond Returns (outliers removed, focus on trend)
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(data_clean['Fed_Rate_Deviation'], data_clean['Bond_Return'], alpha=0.5, s=30, color='steelblue', label='Data')
    # Add linear trend line
    z_linear = np.polyfit(data_clean['Fed_Rate_Deviation'], data_clean['Bond_Return'], 1)
    p_linear = np.poly1d(z_linear)
    x_trend = np.linspace(data_clean['Fed_Rate_Deviation'].min(), data_clean['Fed_Rate_Deviation'].max(), 100)
    ax2.plot(x_trend, p_linear(x_trend), "r-", alpha=0.9, linewidth=3, 
             label=f'Linear Trend (corr={corr_deviation:.3f})')
    # Add polynomial trend (2nd degree) for better fit
    z_poly = np.polyfit(data_clean['Fed_Rate_Deviation'], data_clean['Bond_Return'], 2)
    p_poly = np.poly1d(z_poly)
    ax2.plot(x_trend, p_poly(x_trend), "g--", alpha=0.8, linewidth=2, 
             label='Polynomial Trend')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Fed Rate Deviation from 12M Avg (percentage points)')
    ax2.set_ylabel('Bond Return (%)')
    ax2.set_title(f'Fed Rate Deviation vs {bond_label} Bond Returns (Outliers Removed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Yearly smoothed returns vs Fed Rate Deviation (yearly, outliers removed, focus on trend)
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(data_yearly_clean['Fed_Rate_Deviation'], data_yearly_clean['Bond_Return_Smoothed'], 
                alpha=0.6, s=80, color='purple', label='Yearly Data')
    # Linear trend
    z_linear_y = np.polyfit(data_yearly_clean['Fed_Rate_Deviation'], data_yearly_clean['Bond_Return_Smoothed'], 1)
    p_linear_y = np.poly1d(z_linear_y)
    x_trend_y = np.linspace(data_yearly_clean['Fed_Rate_Deviation'].min(), 
                            data_yearly_clean['Fed_Rate_Deviation'].max(), 100)
    ax3.plot(x_trend_y, p_linear_y(x_trend_y), "r-", alpha=0.9, linewidth=3, 
             label=f'Linear Trend (corr={corr_deviation_smoothed:.3f})')
    # Polynomial trend
    z_poly_y = np.polyfit(data_yearly_clean['Fed_Rate_Deviation'], data_yearly_clean['Bond_Return_Smoothed'], 2)
    p_poly_y = np.poly1d(z_poly_y)
    ax3.plot(x_trend_y, p_poly_y(x_trend_y), "g--", alpha=0.8, linewidth=2, 
             label='Polynomial Trend')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Fed Rate Deviation from 12M Avg (percentage points)')
    ax3.set_ylabel('Smoothed Bond Return (12M avg, %)')
    ax3.set_title(f'Fed Rate Deviation vs Smoothed Bond Returns - Yearly (Outliers Removed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot: Bond Returns when Fed Rate above vs below average
    ax4 = plt.subplot(3, 2, 4)
    box_data = [above_avg['Bond_Return'].values, below_avg['Bond_Return'].values]
    bp = ax4.boxplot(box_data, labels=['Above Avg', 'Below Avg'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax4.set_ylabel('Bond Return (%)')
    ax4.set_title(f'Bond Returns: Fed Rate Above vs Below 12M Average ({bond_label})')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Time series: Fed Rate Deviation over time with trend
    ax5 = plt.subplot(3, 2, 5)
    colors = ['red' if x > 0 else 'green' for x in data_clean['Fed_Rate_Deviation']]
    ax5.bar(data_clean.index, data_clean['Fed_Rate_Deviation'], color=colors, alpha=0.6, width=20)
    # Add rolling average trend line
    trend_window = 12
    data_clean['Fed_Dev_Trend'] = data_clean['Fed_Rate_Deviation'].rolling(window=trend_window).mean()
    ax5.plot(data_clean.index, data_clean['Fed_Dev_Trend'], 'k-', linewidth=2.5, 
             label=f'{trend_window}-Month Moving Average', alpha=0.8)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Deviation (percentage points)')
    ax5.set_title(f'Fed Rate Deviation from 12M Average Over Time ({bond_label})')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Distribution: Bond Returns when Fed above vs below average
    ax6 = plt.subplot(3, 2, 6)
    ax6.hist(above_avg['Bond_Return'], bins=25, alpha=0.6, label='Fed Above Avg', 
             color='red', density=True)
    ax6.hist(below_avg['Bond_Return'], bins=25, alpha=0.6, label='Fed Below Avg', 
             color='green', density=True)
    # Add mean lines
    ax6.axvline(above_avg['Bond_Return'].mean(), color='darkred', linestyle='--', 
                linewidth=2, label=f'Mean Above: {above_avg["Bond_Return"].mean():.2f}%')
    ax6.axvline(below_avg['Bond_Return'].mean(), color='darkgreen', linestyle='--', 
                linewidth=2, label=f'Mean Below: {below_avg["Bond_Return"].mean():.2f}%')
    ax6.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax6.set_xlabel('Bond Return (%)')
    ax6.set_ylabel('Density')
    ax6.set_title(f'Bond Return Distribution: Fed Above vs Below Avg ({bond_label})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{bond_name}_fed_deviation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFed deviation analysis saved to '{output_path}'")
    plt.close()
    
    return corr_deviation, data_clean, data_yearly_clean

# Analyze each bond
print("\n" + "="*60)
corr_1_3, data_1_3 = analyze_bond(data_1_3, '1_3_year', output_folders['1-3_year'], '1-3 Year')

print("\n" + "="*60)
corr_3_7, data_3_7 = analyze_bond(data_3_7, '3_7_year', output_folders['3-7_year'], '3-7 Year')

print("\n" + "="*60)
corr_20_plus, data_20_plus = analyze_bond(data_20_plus, '20_plus_year', output_folders['20_plus_year'], '20+ Year')

# In-depth analysis: Fed rate deviation from 12-month average
print("\n" + "="*60)
print("\n=== FED RATE DEVIATION ANALYSIS ===")
print("\n" + "="*60)
corr_dev_1_3, data_dev_1_3, data_yearly_1_3 = analyze_fed_deviation(data_1_3, '1_3_year', output_folders['1-3_year'], '1-3 Year')

print("\n" + "="*60)
corr_dev_3_7, data_dev_3_7, data_yearly_3_7 = analyze_fed_deviation(data_3_7, '3_7_year', output_folders['3-7_year'], '3-7 Year')

print("\n" + "="*60)
corr_dev_20_plus, data_dev_20_plus, data_yearly_20_plus = analyze_fed_deviation(data_20_plus, '20_plus_year', output_folders['20_plus_year'], '20+ Year')

# Overall summary
print("\n" + "="*60)
print("\n=== OVERALL SUMMARY ===")
print(f"\nCorrelation between 1-3 Year Bond Returns and Fed Rate: {corr_1_3:.4f}")
print(f"Correlation between 3-7 Year Bond Returns and Fed Rate: {corr_3_7:.4f}")
print(f"Correlation between 20+ Year Bond Returns and Fed Rate: {corr_20_plus:.4f}")

print(f"\n=== FED RATE DEVIATION CORRELATIONS ===")
print(f"Correlation between 1-3 Year Bond Returns and Fed Rate Deviation: {corr_dev_1_3:.4f}")
print(f"Correlation between 3-7 Year Bond Returns and Fed Rate Deviation: {corr_dev_3_7:.4f}")
print(f"Correlation between 20+ Year Bond Returns and Fed Rate Deviation: {corr_dev_20_plus:.4f}")

# Create overall comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Comparison scatter plots
axes[0, 0].scatter(data_1_3['Fed_Rate'], data_1_3['Bond_Return'], alpha=0.5, s=30, label='1-3 Year')
z = np.polyfit(data_1_3['Fed_Rate'], data_1_3['Bond_Return'], 1)
p = np.poly1d(z)
axes[0, 0].plot(data_1_3['Fed_Rate'], p(data_1_3['Fed_Rate']), "r--", alpha=0.8, linewidth=2)
axes[0, 0].set_xlabel('Fed Funds Rate (%)')
axes[0, 0].set_ylabel('Bond Return (%)')
axes[0, 0].set_title(f'1-3 Year (corr={corr_1_3:.3f})')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].scatter(data_3_7['Fed_Rate'], data_3_7['Bond_Return'], alpha=0.5, s=30, label='3-7 Year', color='orange')
z = np.polyfit(data_3_7['Fed_Rate'], data_3_7['Bond_Return'], 1)
p = np.poly1d(z)
axes[0, 1].plot(data_3_7['Fed_Rate'], p(data_3_7['Fed_Rate']), "r--", alpha=0.8, linewidth=2)
axes[0, 1].set_xlabel('Fed Funds Rate (%)')
axes[0, 1].set_ylabel('Bond Return (%)')
axes[0, 1].set_title(f'3-7 Year (corr={corr_3_7:.3f})')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].scatter(data_20_plus['Fed_Rate'], data_20_plus['Bond_Return'], alpha=0.5, s=30, label='20+ Year', color='green')
z = np.polyfit(data_20_plus['Fed_Rate'], data_20_plus['Bond_Return'], 1)
p = np.poly1d(z)
axes[1, 0].plot(data_20_plus['Fed_Rate'], p(data_20_plus['Fed_Rate']), "r--", alpha=0.8, linewidth=2)
axes[1, 0].set_xlabel('Fed Funds Rate (%)')
axes[1, 0].set_ylabel('Bond Return (%)')
axes[1, 0].set_title(f'20+ Year (corr={corr_20_plus:.3f})')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Correlation comparison bar chart
axes[1, 1].bar(['1-3 Year', '3-7 Year', '20+ Year'], 
               [corr_1_3, corr_3_7, corr_20_plus], 
               color=['steelblue', 'orange', 'green'], alpha=0.7)
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_ylabel('Correlation with Fed Rate')
axes[1, 1].set_title('Correlation Comparison')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([corr_1_3, corr_3_7, corr_20_plus]):
    axes[1, 1].text(i, v + 0.01 if v >= 0 else v - 0.01, f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')
print("\nOverall comparison saved to 'overall_comparison.png'")
plt.close()

# Validazione teoria Duration: ΔP/P ≈ -Duration × Δr
print("\n" + "="*60)
print("\n=== VALIDAZIONE TEORIA DURATION ===")
print("Teoria: ΔP/P ≈ -Duration × Δr")
print("Se Fed alza tassi di 1%, bond a 10 anni dovrebbero crollare del 10%")
print("\n" + "="*60)

def validate_duration_theory(data, bond_name, output_folder, bond_label, estimated_duration):
    """Valida la teoria della duration: ΔP/P ≈ -Duration × Δr"""
    data = data.copy()
    
    # Calcola variazione mensile dei tassi Fed (in punti percentuali)
    data['Fed_Rate_Change'] = data['Fed_Rate'].diff()
    
    # Calcola variazione percentuale del prezzo del bond
    data['Bond_Price_Change_Pct'] = data['Bond_Price'].pct_change() * 100
    
    # Rimuovi NaN
    data = data.dropna()
    
    # Rimuovi outliers
    data_clean = remove_outliers_iqr(data, ['Fed_Rate_Change', 'Bond_Price_Change_Pct'])
    print(f"\n{bond_label}: Removed {len(data) - len(data_clean)} outliers")
    
    # Calcola valore teorico secondo la teoria: ΔP/P teorico = -Duration × Δr
    data_clean['Theoretical_Change'] = -estimated_duration * data_clean['Fed_Rate_Change']
    
    # Calcola errore (differenza tra teorico e reale)
    data_clean['Error'] = data_clean['Bond_Price_Change_Pct'] - data_clean['Theoretical_Change']
    
    # Calcola duration effettiva dai dati (regressione lineare)
    z2 = np.polyfit(data_clean['Fed_Rate_Change'], data_clean['Bond_Price_Change_Pct'], 1)
    effective_duration = -z2[0]  # Duration effettiva = -slope
    
    # Statistiche
    correlation = data_clean['Bond_Price_Change_Pct'].corr(data_clean['Theoretical_Change'])
    mae = data_clean['Error'].abs().mean()  # Mean Absolute Error
    rmse = np.sqrt((data_clean['Error']**2).mean())  # Root Mean Square Error
    
    print(f"\n{bond_label} (Duration stimata: {estimated_duration} anni):")
    print(f"  Duration effettiva dai dati: {effective_duration:.2f} anni")
    print(f"  Correlazione tra cambiamento reale e teorico: {correlation:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}%")
    print(f"  Root Mean Square Error: {rmse:.4f}%")
    print(f"  Errore medio: {data_clean['Error'].mean():.4f}%")
    print(f"  Std errore: {data_clean['Error'].std():.4f}%")
    
    # Crea visualizzazioni
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Scatter: Cambiamento teorico vs reale
    ax1 = plt.subplot(3, 2, 1)
    ax1.scatter(data_clean['Theoretical_Change'], data_clean['Bond_Price_Change_Pct'], 
                alpha=0.6, s=40, color='steelblue')
    # Linea perfetta (y=x)
    min_val = min(data_clean['Theoretical_Change'].min(), data_clean['Bond_Price_Change_Pct'].min())
    max_val = max(data_clean['Theoretical_Change'].max(), data_clean['Bond_Price_Change_Pct'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Linea perfetta (y=x)', alpha=0.8)
    # Trend line reale
    z = np.polyfit(data_clean['Theoretical_Change'], data_clean['Bond_Price_Change_Pct'], 1)
    p = np.poly1d(z)
    ax1.plot(data_clean['Theoretical_Change'], p(data_clean['Theoretical_Change']), 
             'g-', linewidth=2, label=f'Trend reale (slope={z[0]:.3f})', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Cambiamento Teorico (%) = -Duration × ΔFed_Rate')
    ax1.set_ylabel('Cambiamento Reale Bond Price (%)')
    ax1.set_title(f'Teoria Duration: Teorico vs Reale ({bond_label}, Duration={estimated_duration})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter: Fed Rate Change vs Bond Price Change con linea teorica
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(data_clean['Fed_Rate_Change'], data_clean['Bond_Price_Change_Pct'], 
                alpha=0.6, s=40, color='steelblue', label='Dati reali')
    # Linea teorica
    x_theory = np.linspace(data_clean['Fed_Rate_Change'].min(), 
                          data_clean['Fed_Rate_Change'].max(), 100)
    y_theory = -estimated_duration * x_theory
    ax2.plot(x_theory, y_theory, 'r--', linewidth=3, 
             label=f'Teoria: -{estimated_duration} × ΔFed_Rate', alpha=0.8)
    # Trend line reale (usa z2 già calcolato)
    p2 = np.poly1d(z2)
    ax2.plot(data_clean['Fed_Rate_Change'], p2(data_clean['Fed_Rate_Change']), 
             'g-', linewidth=2, label=f'Trend reale (slope={z2[0]:.2f}, Duration effettiva={effective_duration:.2f})', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle=':', linewidth=0.5)
    ax2.set_xlabel('Cambiamento Fed Rate (punti percentuali)')
    ax2.set_ylabel('Cambiamento Bond Price (%)')
    ax2.set_title(f'Fed Rate Change vs Bond Price Change ({bond_label})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribuzione errori
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(data_clean['Error'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax3.axvline(x=data_clean['Error'].mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Media: {data_clean["Error"].mean():.3f}%')
    ax3.set_xlabel('Errore (%) = Reale - Teorico')
    ax3.set_ylabel('Frequenza')
    ax3.set_title(f'Distribuzione Errori ({bond_label})')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time series: Fed Rate Change, Bond Change Reale, Bond Change Teorico
    ax4 = plt.subplot(3, 2, 4)
    ax4_twin = ax4.twinx()
    ax4.plot(data_clean.index, data_clean['Fed_Rate_Change'], 'b-', alpha=0.6, 
             linewidth=1.5, label='Fed Rate Change')
    ax4_twin.plot(data_clean.index, data_clean['Bond_Price_Change_Pct'], 'g-', alpha=0.7, 
                  linewidth=2, label='Bond Change Reale')
    ax4_twin.plot(data_clean.index, data_clean['Theoretical_Change'], 'r--', alpha=0.7, 
                  linewidth=2, label='Bond Change Teorico')
    ax4.axhline(y=0, color='b', linestyle=':', linewidth=0.5)
    ax4_twin.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Fed Rate Change (pp)', color='b')
    ax4_twin.set_ylabel('Bond Price Change (%)', color='k')
    ax4.set_title(f'Confronto Reale vs Teorico nel Tempo ({bond_label})')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter: Errore vs Fed Rate Change
    ax5 = plt.subplot(3, 2, 5)
    ax5.scatter(data_clean['Fed_Rate_Change'], data_clean['Error'], alpha=0.6, s=40, color='orange')
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax5.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    z3 = np.polyfit(data_clean['Fed_Rate_Change'], data_clean['Error'], 1)
    p3 = np.poly1d(z3)
    ax5.plot(data_clean['Fed_Rate_Change'], p3(data_clean['Fed_Rate_Change']), 
             'r-', linewidth=2, label=f'Trend errore (slope={z3[0]:.3f})', alpha=0.8)
    ax5.set_xlabel('Fed Rate Change (punti percentuali)')
    ax5.set_ylabel('Errore (%)')
    ax5.set_title(f'Errore vs Fed Rate Change ({bond_label})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Rolling correlation tra teorico e reale
    ax6 = plt.subplot(3, 2, 6)
    window = 12
    rolling_corr = data_clean['Bond_Price_Change_Pct'].rolling(window=window).corr(
        data_clean['Theoretical_Change'])
    ax6.plot(data_clean.index, rolling_corr, linewidth=2, color='steelblue')
    ax6.axhline(y=correlation, color='r', linestyle='--', linewidth=1, 
                label=f'Correlazione totale: {correlation:.3f}')
    ax6.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Rolling Correlation (12-month)')
    ax6.set_title(f'Rolling Correlation: Reale vs Teorico ({bond_label})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{bond_name}_duration_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDuration validation saved to '{output_path}'")
    plt.close()
    
    return correlation, mae, rmse, data_clean, effective_duration

# Stima duration approssimative (per bond zero-coupon, duration ≈ maturity)
# Per bond con coupon, duration è leggermente inferiore alla maturity
duration_estimates = {
    '1-3_year': 2.0,  # Media tra 1 e 3 anni
    '3-7_year': 5.0,  # Media tra 3 e 7 anni
    '20_plus_year': 20.0  # Per bond 20+ anni
}

# Ricarica i dati originali per l'analisi
data_1_3_orig = fed_rate.join(bond_1_3, how='inner')
data_3_7_orig = fed_rate.join(bond_3_7, how='inner')
data_20_plus_orig = fed_rate.join(bond_20_plus, how='inner')

data_1_3_orig.columns = ['Fed_Rate', 'Bond_Price']
data_3_7_orig.columns = ['Fed_Rate', 'Bond_Price']
data_20_plus_orig.columns = ['Fed_Rate', 'Bond_Price']

print("\n" + "="*60)
corr_dur_1_3, mae_1_3, rmse_1_3, data_dur_1_3, eff_dur_1_3 = validate_duration_theory(
    data_1_3_orig, '1_3_year', output_folders['1-3_year'], '1-3 Year', duration_estimates['1-3_year'])

print("\n" + "="*60)
corr_dur_3_7, mae_3_7, rmse_3_7, data_dur_3_7, eff_dur_3_7 = validate_duration_theory(
    data_3_7_orig, '3_7_year', output_folders['3-7_year'], '3-7 Year', duration_estimates['3-7_year'])

print("\n" + "="*60)
corr_dur_20_plus, mae_20_plus, rmse_20_plus, data_dur_20_plus, eff_dur_20_plus = validate_duration_theory(
    data_20_plus_orig, '20_plus_year', output_folders['20_plus_year'], '20+ Year', duration_estimates['20_plus_year'])

# Summary finale
print("\n" + "="*60)
print("\n=== RIEPILOGO VALIDAZIONE TEORIA DURATION ===")
print(f"\n1-3 Year:")
print(f"  Duration stimata: {duration_estimates['1-3_year']} anni")
print(f"  Duration effettiva dai dati: {eff_dur_1_3:.2f} anni")
print(f"  Correlazione teorico vs reale: {corr_dur_1_3:.4f}")
print(f"  MAE: {mae_1_3:.4f}%, RMSE: {rmse_1_3:.4f}%")

print(f"\n3-7 Year:")
print(f"  Duration stimata: {duration_estimates['3-7_year']} anni")
print(f"  Duration effettiva dai dati: {eff_dur_3_7:.2f} anni")
print(f"  Correlazione teorico vs reale: {corr_dur_3_7:.4f}")
print(f"  MAE: {mae_3_7:.4f}%, RMSE: {rmse_3_7:.4f}%")

print(f"\n20+ Year:")
print(f"  Duration stimata: {duration_estimates['20_plus_year']} anni")
print(f"  Duration effettiva dai dati: {eff_dur_20_plus:.2f} anni")
print(f"  Correlazione teorico vs reale: {corr_dur_20_plus:.4f}")
print(f"  MAE: {mae_20_plus:.4f}%, RMSE: {rmse_20_plus:.4f}%")

print("\n" + "="*60)
print("Analysis complete!")
