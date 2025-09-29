import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_leveraged_data():
    """Load MSCI World 2X leveraged data from file"""
    # Read the text file with tab-separated values
    df = pd.read_csv('MSCIWORLD2X.xls', sep='\t', skiprows=3)
    df.columns = ['Date', 'Price']
    
    # Remove rows that don't contain valid dates (footer text)
    df = df[df['Date'].str.contains(r'\d{1,2}/\d{1,2}/\d{4}', na=False)]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def download_msci_world():
    """Download MSCI World data from Yahoo Finance"""
    # Try different tickers for MSCI World
    tickers = ['URTH', 'ACWI', 'VT']  # Common MSCI World ETFs
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2000-12-29', progress=False)
            if not data.empty:
                print(f"Successfully downloaded {ticker}")
                # Handle multi-level columns - yfinance now uses 'Close' by default with auto_adjust=True
                if isinstance(data.columns, pd.MultiIndex):
                    close_price = data[('Close', ticker)]
                else:
                    close_price = data['Close']
                return close_price.to_frame(name='Price')
        except:
            continue
    
    # If no ETF works, try to get synthetic data based on SPY
    print("Using SPY as proxy for MSCI World")
    data = yf.download('SPY', start='2000-12-29', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close_price = data[('Close', 'SPY')]
    else:
        close_price = data['Close']
    return close_price.to_frame(name='Price')

def calculate_metrics(returns, risk_free_rate=0.02):
    """Calculate performance metrics"""
    # Annualized metrics
    cagr = (1 + returns.mean()) ** 252 - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else np.nan
    
    return {
        'CAGR': cagr * 100,
        'Volatility': volatility * 100,
        'Sharpe': sharpe,
        'Sortino': sortino
    }

def calculate_drawdown(prices):
    """Calculate drawdown series"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown

def calculate_rolling_returns(prices, windows=[30, 90, 252, 504, 756, 1008]):
    """Calculate rolling returns for different periods"""
    rolling_returns = {}
    for window in windows:
        rolling_returns[f'{window}d'] = prices.pct_change(window)
    return rolling_returns

def main():
    print("Loading MSCI World 2X leveraged data...")
    leveraged_data = load_leveraged_data()
    
    print("Downloading MSCI World data from Yahoo Finance...")
    msci_world_data = download_msci_world()
    
    # Align dates
    start_date = max(leveraged_data.index.min(), msci_world_data.index.min())
    end_date = min(leveraged_data.index.max(), msci_world_data.index.max())
    
    leveraged_data = leveraged_data.loc[start_date:end_date]
    msci_world_data = msci_world_data.loc[start_date:end_date]
    
    print(f"Analysis period: {start_date.date()} to {end_date.date()}")
    
    # Calculate returns
    leveraged_returns = leveraged_data['Price'].pct_change().dropna()
    msci_returns = msci_world_data['Price'].pct_change().dropna()
    
    # Calculate metrics
    leveraged_metrics = calculate_metrics(leveraged_returns)
    msci_metrics = calculate_metrics(msci_returns)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'MSCI World 2X': leveraged_metrics,
        'MSCI World': msci_metrics
    }).T
    
    # 1. Normalized price evolution plot
    leveraged_normalized = leveraged_data['Price'] / leveraged_data['Price'].iloc[0] * 100
    msci_normalized = msci_world_data['Price'] / msci_world_data['Price'].iloc[0] * 100
    
    plt.figure(figsize=(15, 8))
    plt.plot(leveraged_normalized.index, leveraged_normalized, 
             label='MSCI World 2X', linewidth=2, color='#FF6347')
    plt.plot(msci_normalized.index, msci_normalized, 
             label='MSCI World', linewidth=2, color='#2E8B57')
    
    plt.title('Normalized Price Evolution: MSCI World vs MSCI World 2X\n(Base = 100 at start date)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Base = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('normalized_price_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Bar chart for performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Metrics Comparison: MSCI World vs MSCI World 2X', fontsize=16, fontweight='bold')
    
    metrics = ['CAGR', 'Volatility', 'Sharpe', 'Sortino']
    colors = ['#2E8B57', '#FF6347']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [comparison_df.loc['MSCI World', metric], comparison_df.loc['MSCI World 2X', metric]]
        bars = ax.bar(['MSCI World', 'MSCI World 2X'], values, color=colors, alpha=0.8)
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_ylabel(metric)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Drawdown plot
    leveraged_drawdown = calculate_drawdown(leveraged_data['Price'])
    msci_drawdown = calculate_drawdown(msci_world_data['Price'])
    
    plt.figure(figsize=(15, 8))
    plt.plot(leveraged_drawdown.index, leveraged_drawdown * 100, 
             label='MSCI World 2X', linewidth=2, color='#FF6347')
    plt.plot(msci_drawdown.index, msci_drawdown * 100, 
             label='MSCI World', linewidth=2, color='#2E8B57')
    
    plt.fill_between(leveraged_drawdown.index, leveraged_drawdown * 100, 0, 
                     alpha=0.3, color='#FF6347')
    plt.fill_between(msci_drawdown.index, msci_drawdown * 100, 0, 
                     alpha=0.3, color='#2E8B57')
    
    plt.title('Drawdown Comparison: MSCI World vs MSCI World 2X', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('drawdown_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Rolling returns plot with longer intervals
    leveraged_rolling = calculate_rolling_returns(leveraged_data['Price'])
    msci_rolling = calculate_rolling_returns(msci_world_data['Price'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Rolling Returns Comparison: MSCI World vs MSCI World 2X', fontsize=16, fontweight='bold')
    
    windows = [30, 90, 252, 504, 756, 1008]  # 1M, 3M, 1Y, 2Y, 3Y, 4Y
    window_labels = ['1 Month', '3 Months', '1 Year', '2 Years', '3 Years', '4 Years']
    
    for i, (window, label) in enumerate(zip(windows, window_labels)):
        ax = axes[i//3, i%3]
        
        leveraged_ret = leveraged_rolling[f'{window}d'] * 100
        msci_ret = msci_rolling[f'{window}d'] * 100
        
        ax.plot(leveraged_ret.index, leveraged_ret, 
                label='MSCI World 2X', linewidth=1.5, color='#FF6347', alpha=0.8)
        ax.plot(msci_ret.index, msci_ret, 
                label='MSCI World', linewidth=1.5, color='#2E8B57', alpha=0.8)
        
        ax.set_title(f'{label} Rolling Returns', fontweight='bold')
        ax.set_ylabel('Returns (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('rolling_returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(comparison_df.round(2))
    
    print(f"\nAnalysis completed! Charts saved as:")
    print("- normalized_price_evolution.png")
    print("- performance_metrics_comparison.png")
    print("- drawdown_comparison.png") 
    print("- rolling_returns_comparison.png")

if __name__ == "__main__":
    main()
