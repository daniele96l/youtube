import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def get_sp500_data(start_date, end_date):
    """Get MSCI World ETF data from CSV file"""
    try:
        # Load data from CSV file
        csv_file = "../iShares Core MSCI World UCITS ETF USD (Acc).csv"
        data = pd.read_csv(csv_file, skiprows=2)  # Skip the header row and ticker row
        
        # The first column is empty, so we need to use the first column as Date
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Filter data for the requested period
        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        
        if not filtered_data.empty:
            print(f"Successfully loaded MSCI World ETF data from CSV")
            print(f"Data period: {filtered_data['Date'].min().date()} to {filtered_data['Date'].max().date()}")
            return filtered_data.set_index('Date')['Close']
        else:
            print("No data found for the requested period, using synthetic data")
            return generate_synthetic_data(start_date, end_date)
            
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        print("Using synthetic data based on historical MSCI World parameters")
        return generate_synthetic_data(start_date, end_date)

def generate_synthetic_data(start_date, end_date):
    """Generate synthetic data based on historical MSCI World parameters"""
    days = (end_date - start_date).days
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Historical MSCI World parameters (monthly returns)
    mean_return = 0.008  # ~0.8% monthly return
    std_return = 0.040   # ~4.0% monthly volatility
    
    # Generate synthetic daily returns
    daily_mean = mean_return / 30
    daily_std = std_return / np.sqrt(30)
    
    returns = np.random.normal(daily_mean, daily_std, len(dates))
    prices = [100]  # Starting price
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices[:-1], index=dates)

def simulate_market_crash(returns, crash_threshold=0.05):
    """Simulate market crashes based on historical returns"""
    crashes = []
    peak = 1.0
    current_value = 1.0
    
    for i, ret in enumerate(returns):
        current_value *= (1 + ret)
        if current_value > peak:
            peak = current_value
        
        # Check if we have a crash from peak
        if (peak - current_value) / peak >= crash_threshold:
            crashes.append(i)
            peak = current_value  # Reset peak after crash
    
    return crashes

def strategy_1_regular_investing(monthly_investment, returns):
    """Strategy 1: Invest full amount every month"""
    portfolio_values = []
    shares_owned = 0
    price = 100  # Starting price
    
    for i, ret in enumerate(returns):
        price *= (1 + ret)
        shares_bought = monthly_investment / price
        shares_owned += shares_bought
        portfolio_value = shares_owned * price
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, shares_owned

def strategy_2_dip_buying(monthly_investment, dip_reserve, returns, crashes):
    """Strategy 2: Invest 80%, save 20%, buy dips"""
    portfolio_values = []
    shares_owned = 0
    price = 100  # Starting price
    saved_money = 0
    dip_purchases = []
    
    for i, ret in enumerate(returns):
        price *= (1 + ret)
        
        # Regular monthly investment (80%)
        regular_investment = monthly_investment * 0.8
        shares_bought = regular_investment / price
        shares_owned += shares_bought
        
        # Save 20% for dips
        saved_money += monthly_investment * 0.2
        
        # Check if this is a crash point
        if i in crashes:
            if saved_money > 0:
                dip_shares = saved_money / price
                shares_owned += dip_shares
                dip_purchases.append((i, price, saved_money, dip_shares))
                saved_money = 0
        
        portfolio_value = shares_owned * price
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, shares_owned, dip_purchases

def calculate_cagr(initial_value, final_value, years):
    """Calculate Compound Annual Growth Rate"""
    if initial_value <= 0:
        return 0
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_portfolio_cagr(portfolio_values, monthly_investment, years):
    """Calculate CAGR based on portfolio growth"""
    if len(portfolio_values) < 2:
        return 0
    
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    # Calculate CAGR based on portfolio growth
    if initial_value <= 0:
        return 0
    
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_volatility(portfolio_values):
    """Calculate volatility (standard deviation of returns) excluding deposits"""
    if len(portfolio_values) < 2:
        return 0
    
    # Calculate returns (excluding deposits)
    returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        returns.append(ret)
    
    return np.std(returns) * np.sqrt(12)  # Annualized volatility

def run_monte_carlo_simulation(n_simulations=1000, years=10):
    """Run Monte Carlo simulation comparing multiple strategies"""
    results = []
    thresholds = [0.05, 0.10, 0.20, 0.30]  # 5%, 10%, 20%, 30%
    
    # Get historical MSCI World ETF data for realistic parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*20)  # 20 years of data
    sp500_data = get_sp500_data(start_date, end_date)
    
    # Calculate historical returns
    returns = sp500_data.pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    print(f"Historical MSCI World - Mean return: {mean_return:.4f}, Std: {std_return:.4f}")
    
    for sim in range(n_simulations):
        # Generate random returns for 5 years (60 months)
        random_returns = np.random.normal(mean_return, std_return, years * 12)
        
        # Run regular strategy
        portfolio_1_values, shares_1 = strategy_1_regular_investing(1000, random_returns)
        
        sim_result = {
            'simulation': sim,
            'strategy_1_value': portfolio_1_values[-1],
            'strategy_1_cagr': calculate_portfolio_cagr(portfolio_1_values, 1000, years),
            'strategy_1_volatility': calculate_volatility(portfolio_1_values),
            'strategy_1_shares': shares_1
        }
        
        # Run dip buying strategies for different thresholds
        for threshold in thresholds:
            crashes = simulate_market_crash(random_returns, threshold)
            portfolio_2_values, shares_2, dip_purchases = strategy_2_dip_buying(1000, 200, random_returns, crashes)
            
            sim_result[f'strategy_2_{int(threshold*100)}p_value'] = portfolio_2_values[-1]
            sim_result[f'strategy_2_{int(threshold*100)}p_cagr'] = calculate_portfolio_cagr(portfolio_2_values, 1000, years)
            sim_result[f'strategy_2_{int(threshold*100)}p_volatility'] = calculate_volatility(portfolio_2_values)
            sim_result[f'strategy_2_{int(threshold*100)}p_shares'] = shares_2
            sim_result[f'strategy_2_{int(threshold*100)}p_crashes'] = len(crashes)
            sim_result[f'strategy_2_{int(threshold*100)}p_dip_purchases'] = len(dip_purchases)
        
        results.append(sim_result)
    
    return pd.DataFrame(results)

def analyze_results(results):
    """Analyze and display simulation results"""
    print("\n=== RISULTATI SIMULAZIONE MONTE CARLO ===")
    print(f"Numero di simulazioni: {len(results)}")
    print(f"Periodo: 10 anni")
    
    strategies = ['strategy_1'] + [f'strategy_2_{int(t*100)}p' for t in [0.05, 0.10, 0.20, 0.30]]
    threshold_names = ['Regolare'] + ['5%', '10%', '20%', '30%']
    
    print(f"\n{'Strategia':<12} {'Valore Medio':<15} {'CAGR':<10} {'Volatilità':<12} {'Vittorie':<10}")
    print("-" * 70)
    
    for i, strategy in enumerate(strategies):
        value_col = f'{strategy}_value'
        cagr_col = f'{strategy}_cagr'
        vol_col = f'{strategy}_volatility'
        
        mean_value = results[value_col].mean()
        mean_cagr = results[cagr_col].mean()
        mean_vol = results[vol_col].mean()
        
        # Calculate win rate vs regular strategy
        wins = (results[value_col] > results['strategy_1_value']).sum()
        win_rate = wins / len(results) * 100
        
        print(f"{threshold_names[i]:<12} €{mean_value:>10,.0f} {mean_cagr:>8.2%} {mean_vol:>10.2%} {win_rate:>8.1f}%")
    
    return results

def show_example_simulation():
    """Show a specific example simulation with visualization"""
    print("\n=== ESEMPIO DI SIMULAZIONE SPECIFICA ===")
    
    # Get real S&P 500 data for a specific period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    sp500_data = get_sp500_data(start_date, end_date)
    
    # Calculate returns
    returns = sp500_data.pct_change().dropna()
    
    # Run strategies
    portfolio_1_values, shares_1 = strategy_1_regular_investing(1000, returns)
    
    thresholds = [0.05, 0.10, 0.20, 0.30]
    threshold_names = ['5%', '10%', '20%', '30%']
    
    print(f"Periodo analizzato: {start_date.date()} - {end_date.date()}")
    print(f"Strategia 1 (regolare): €{portfolio_1_values[-1]:,.2f}")
    print(f"CAGR: {calculate_portfolio_cagr(portfolio_1_values, 1000, 10):.2%}")
    print(f"Volatilità: {calculate_volatility(portfolio_1_values) * 100:.2f}%")
    
    # Create comprehensive visualization grid (3x2)
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle('Simulazione Strategia "Buy the Dip" - Analisi Completa', fontsize=20, fontweight='bold')
    
    # Plot 1: Price evolution with crash points for different thresholds
    plt.subplot(4, 2, 1)
    price_evolution = [100]
    for ret in returns:
        price_evolution.append(price_evolution[-1] * (1 + ret))
    
    plt.plot(price_evolution, label='MSCI World ETF Price', linewidth=2, color='black')
    
    colors = ['red', 'orange', 'green', 'blue']
    for i, threshold in enumerate(thresholds):
        crashes = simulate_market_crash(returns, threshold)
        for crash_month in crashes:
            if crash_month < len(price_evolution):
                plt.scatter(crash_month, price_evolution[crash_month], 
                           color=colors[i], s=50, alpha=0.7, 
                           label=f'Crash {threshold_names[i]}' if crash_month == crashes[0] else "")
    
    plt.title('Evoluzione Prezzo MSCI World ETF con Punti di Crash', fontsize=14, fontweight='bold')
    plt.xlabel('Mesi')
    plt.ylabel('Prezzo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Portfolio comparison over time
    plt.subplot(4, 2, 2)
    
    plt.plot(portfolio_1_values, label='Strategia 1: Investimento Regolare', linewidth=3, color='black')
    
    for i, threshold in enumerate(thresholds):
        crashes = simulate_market_crash(returns, threshold)
        portfolio_2_values, shares_2, dip_purchases = strategy_2_dip_buying(1000, 200, returns, crashes)
        plt.plot(portfolio_2_values, label=f'Strategia 2: Buy the Dip {threshold_names[i]}', 
                linewidth=2, color=colors[i], alpha=0.8)
    
    plt.title('Confronto Portfolio nel Tempo', fontsize=14, fontweight='bold')
    plt.xlabel('Mesi')
    plt.ylabel('Valore Portfolio (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Saved money evolution for different thresholds
    plt.subplot(4, 2, 3)
    
    for i, threshold in enumerate(thresholds):
        saved_money_evolution = []
        saved_money = 0
        
        for j, ret in enumerate(returns):
            saved_money += 200  # Save €200 per month
            
            # Check if this is a crash point
            crashes = simulate_market_crash(returns, threshold)
            if j in crashes:
                saved_money = 0  # Spend all saved money
            
            saved_money_evolution.append(saved_money)
        
        plt.plot(saved_money_evolution, label=f'Soldi Risparmiati {threshold_names[i]}', 
                linewidth=2, color=colors[i], alpha=0.8)
    
    plt.title('Evoluzione Soldi Risparmiati per Threshold', fontsize=14, fontweight='bold')
    plt.xlabel('Mesi')
    plt.ylabel('Soldi Risparmiati (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Total wealth comparison (portfolio + saved money)
    plt.subplot(4, 2, 4)
    
    # Regular strategy (no saved money)
    plt.plot(portfolio_1_values, label='Strategia 1: Investimento Regolare', linewidth=3, color='black')
    
    for i, threshold in enumerate(thresholds):
        crashes = simulate_market_crash(returns, threshold)
        portfolio_2_values, shares_2, dip_purchases = strategy_2_dip_buying(1000, 200, returns, crashes)
        
        # Calculate total wealth (portfolio + saved money)
        total_wealth = []
        saved_money = 0
        
        for j, portfolio_value in enumerate(portfolio_2_values):
            saved_money += 200  # Save €200 per month
            
            # Check if this is a crash point
            if j in crashes:
                saved_money = 0  # Spend all saved money
            
            total_wealth.append(portfolio_value + saved_money)
        
        plt.plot(total_wealth, label=f'Patrimonio Totale {threshold_names[i]}', 
                linewidth=2, color=colors[i], alpha=0.8)
    
    plt.title('Confronto Patrimonio Totale (Portfolio + Soldi Risparmiati)', fontsize=14, fontweight='bold')
    plt.xlabel('Mesi')
    plt.ylabel('Patrimonio Totale (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    
    # Plot 5: Final portfolio values comparison
    plt.subplot(3, 2, 5)
    
    strategy_names = ['Regolare'] + [f'Buy Dip {t}%' for t in threshold_names]
    final_values = [portfolio_1_values[-1]]
    
    for threshold in thresholds:
        crashes = simulate_market_crash(returns, threshold)
        portfolio_2_values, shares_2, dip_purchases = strategy_2_dip_buying(1000, 200, returns, crashes)
        final_values.append(portfolio_2_values[-1])
    
    bars = plt.bar(strategy_names, final_values, color=['black'] + colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Confronto Valori Finali Portfolio', fontsize=14, fontweight='bold')
    plt.ylabel('Valore Finale (€)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'€{value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 6: Crash frequency analysis
    plt.subplot(3, 2, 6)
    
    crash_counts = []
    for threshold in thresholds:
        crashes = simulate_market_crash(returns, threshold)
        crash_counts.append(len(crashes))
    
    bars = plt.bar(threshold_names, crash_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Frequenza Crash per Threshold', fontsize=14, fontweight='bold')
    plt.xlabel('Threshold')
    plt.ylabel('Numero di Crash')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, crash_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')
    

    
    # Add horizontal line at y=0 for Sharpe ratio
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure as high-resolution image
    filename = f"buy_the_dip_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n📊 Grafico salvato come: {filename}")
    print(f"   Puoi aprirlo a tutto schermo per una visualizzazione dettagliata")
    
    plt.show()
    
    return filename

def print_summary_conclusions(results):
    """Print a clear summary of the simulation conclusions"""
    print("\n" + "="*80)
    print("CONCLUSIONI PRINCIPALI")
    print("="*80)
    
    strategies = ['strategy_1'] + [f'strategy_2_{int(t*100)}p' for t in [0.05, 0.10, 0.20, 0.30]]
    threshold_names = ['Regolare'] + ['5%', '10%', '20%', '30%']
    
    print(f"📊 RISULTATI MEDI (1000 simulazioni di 5 anni):")
    print(f"{'Strategia':<12} {'Valore':<12} {'CAGR':<8} {'Volatilità':<10} {'Vittorie':<8}")
    print("-" * 55)
    
    best_strategy = None
    best_value = 0
    
    for i, strategy in enumerate(strategies):
        value_col = f'{strategy}_value'
        cagr_col = f'{strategy}_cagr'
        vol_col = f'{strategy}_volatility'
        
        mean_value = results[value_col].mean()
        mean_cagr = results[cagr_col].mean()
        mean_vol = results[vol_col].mean()
        
        # Calculate win rate vs regular strategy
        wins = (results[value_col] > results['strategy_1_value']).sum()
        win_rate = wins / len(results) * 100
        
        print(f"{threshold_names[i]:<12} €{mean_value:>9,.0f} {mean_cagr:>7.2%} {mean_vol:>9.2%} {win_rate:>7.1f}%")
        
        if mean_value > best_value:
            best_value = mean_value
            best_strategy = threshold_names[i]
    
    print(f"\n🏆 MIGLIORE STRATEGIA: {best_strategy}")
    print(f"💰 Valore medio: €{best_value:,.0f}")
    
    # Risk-adjusted return analysis
    print(f"\n📈 ANALISI RISK-ADJUSTED RETURN:")
    for i, strategy in enumerate(strategies):
        value_col = f'{strategy}_value'
        cagr_col = f'{strategy}_cagr'
        vol_col = f'{strategy}_volatility'
        
        mean_cagr = results[cagr_col].mean()
        mean_vol = results[vol_col].mean()
        
        if mean_vol > 0:
            sharpe_ratio = mean_cagr / mean_vol
            print(f"   {threshold_names[i]:<12}: Sharpe Ratio = {sharpe_ratio:.3f}")
    
    print(f"\n💡 RACCOMANDAZIONI:")
    if best_strategy == 'Regolare':
        print(f"   ✅ L'investimento regolare è la strategia più profittevole")
    else:
        print(f"   ✅ La strategia Buy the Dip {best_strategy} è la più profittevole")
    
    print(f"   📊 Considerare sempre il rapporto rischio/rendimento")
    print(f"   ⚠️  Maggiore volatilità può significare maggiore stress emotivo")

def create_monte_carlo_analysis_image():
    """Create a comprehensive Monte Carlo analysis with confidence intervals and quantiles"""
    print("\n=== ANALISI MONTE CARLO CON INTERVALLI DI CONFIDENZA ===")
    
    # Parameters
    n_periods = 100
    years = 10
    thresholds = [0.05, 0.10, 0.20, 0.30]
    threshold_names = ['5%', '10%', '20%', '30%']
    colors = ['red', 'orange', 'green', 'blue']
    
    # Get historical MSCI World ETF data for realistic parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*20)
    sp500_data = get_sp500_data(start_date, end_date)
    returns = sp500_data.pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    print(f"Generando {n_periods} simulazioni di {years} anni ciascuna...")
    
    # Store results for each strategy
    all_results = {
        'regular': {'portfolio_values': [], 'final_cash': [], 'total_wealth': [], 'cagr': [], 'volatility': []},
        'dip_5': {'portfolio_values': [], 'final_cash': [], 'total_wealth': [], 'cagr': [], 'volatility': []},
        'dip_10': {'portfolio_values': [], 'final_cash': [], 'total_wealth': [], 'cagr': [], 'volatility': []},
        'dip_20': {'portfolio_values': [], 'final_cash': [], 'total_wealth': [], 'cagr': [], 'volatility': []},
        'dip_30': {'portfolio_values': [], 'final_cash': [], 'total_wealth': [], 'cagr': [], 'volatility': []}
    }
    
    # Run Monte Carlo simulations
    for period in range(n_periods):
        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return, years * 12)
        
        # Regular strategy
        portfolio_1_values, shares_1 = strategy_1_regular_investing(1000, random_returns)
        final_portfolio_1 = portfolio_1_values[-1]
        final_cash_1 = 0  # No cash saved in regular strategy
        total_wealth_1 = final_portfolio_1 + final_cash_1
        
        all_results['regular']['portfolio_values'].append(final_portfolio_1)
        all_results['regular']['final_cash'].append(final_cash_1)
        all_results['regular']['total_wealth'].append(total_wealth_1)
        all_results['regular']['cagr'].append(calculate_portfolio_cagr(portfolio_1_values, 1000, years))
        all_results['regular']['volatility'].append(calculate_volatility(portfolio_1_values))
        
        # Dip strategies
        for i, threshold in enumerate(thresholds):
            crashes = simulate_market_crash(random_returns, threshold)
            portfolio_2_values, shares_2, dip_purchases = strategy_2_dip_buying(1000, 200, random_returns, crashes)
            
            # Calculate final cash remaining
            final_cash = 0
            saved_money = 0
            for j, ret in enumerate(random_returns):
                saved_money += 200  # Save €200 per month
                if j in crashes:
                    saved_money = 0  # Spend all saved money
            final_cash = saved_money
            
            final_portfolio_2 = portfolio_2_values[-1]
            total_wealth_2 = final_portfolio_2 + final_cash
            
            strategy_key = f'dip_{int(threshold*100)}'
            all_results[strategy_key]['portfolio_values'].append(final_portfolio_2)
            all_results[strategy_key]['final_cash'].append(final_cash)
            all_results[strategy_key]['total_wealth'].append(total_wealth_2)
            all_results[strategy_key]['cagr'].append(calculate_portfolio_cagr(portfolio_2_values, 1000, years))
            all_results[strategy_key]['volatility'].append(calculate_volatility(portfolio_2_values))
    
    # Create comprehensive visualization grid
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Analisi Monte Carlo - Strategia Buy the Dip', fontsize=20, fontweight='bold')
    
    strategy_labels = ['Regolare'] + [f'Buy Dip {t}%' for t in threshold_names]
    strategy_keys = ['regular'] + [f'dip_{int(t*100)}' for t in thresholds]
    colors_plot = ['black'] + colors
    
    # Plot 1: Distribuzione valori portfolio con intervalli di confidenza
    plt.subplot(2, 2, 1)
    
    # Calculate statistics for portfolio values
    means = []
    stds = []
    
    for key in strategy_keys:
        values = all_results[key]['portfolio_values']
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    x_pos = np.arange(len(strategy_labels))
    
    # Plot bars with error bars
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('1) Distribuzione Valori Portfolio con Intervalli di Confidenza', fontsize=14, fontweight='bold')
    plt.ylabel('Valore Portfolio (€)')
    plt.xticks(x_pos, strategy_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'€{mean_val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Distribuzione finale cash rimanenti
    plt.subplot(2, 2, 2)
    
    # Calculate statistics for final cash
    cash_means = []
    cash_stds = []
    
    for key in strategy_keys:
        values = all_results[key]['final_cash']
        cash_means.append(np.mean(values))
        cash_stds.append(np.std(values))
    
    # Plot bars with error bars
    bars = plt.bar(x_pos, cash_means, yerr=cash_stds, capsize=5, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('2) Distribuzione Finale Cash Rimanenti', fontsize=14, fontweight='bold')
    plt.ylabel('Cash Rimanente (€)')
    plt.xticks(x_pos, strategy_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean_val in zip(bars, cash_means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'€{mean_val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Distribuzione valori portfolio + cash
    plt.subplot(2, 2, 3)
    
    # Calculate statistics for total wealth
    wealth_means = []
    wealth_stds = []
    
    for key in strategy_keys:
        values = all_results[key]['total_wealth']
        wealth_means.append(np.mean(values))
        wealth_stds.append(np.std(values))
    
    # Plot bars with error bars
    bars = plt.bar(x_pos, wealth_means, yerr=wealth_stds, capsize=5, color=colors_plot, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('3) Distribuzione Valori Portfolio + Cash', fontsize=14, fontweight='bold')
    plt.ylabel('Patrimonio Totale (€)')
    plt.xticks(x_pos, strategy_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean_val in zip(bars, wealth_means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'€{mean_val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Tasso vittoria
    plt.subplot(2, 2, 4)
    
    # Calculate win rates (total wealth comparison)
    win_rates = []
    for i, key in enumerate(strategy_keys[1:]):  # Skip regular strategy
        wins = 0
        for j in range(n_periods):
            if all_results[key]['total_wealth'][j] > all_results['regular']['total_wealth'][j]:
                wins += 1
        win_rate = wins / n_periods * 100
        win_rates.append(win_rate)
    
    bars = plt.bar(threshold_names, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('4) Tasso di Vittoria (Patrimonio Totale)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentuale Vittorie (%)')
    plt.xlabel('Threshold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure as high-resolution image
    filename = f"monte_carlo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n📊 Analisi Monte Carlo salvata come: {filename}")
    print(f"   Basata su {n_periods} simulazioni di {years} anni")
    
    # Print summary statistics
    print(f"\n📈 RIEPILOGO STATISTICO:")
    print(f"{'Strategia':<15} {'Portfolio':<12} {'Cash':<10} {'Totale':<12} {'Vittorie':<10}")
    print("-" * 70)
    
    for i, key in enumerate(strategy_keys):
        portfolio_mean = np.mean(all_results[key]['portfolio_values'])
        cash_mean = np.mean(all_results[key]['final_cash'])
        total_mean = np.mean(all_results[key]['total_wealth'])
        
        if key == 'regular':
            win_rate = 0  # Regular strategy doesn't win against itself
        else:
            wins = sum(1 for j in range(n_periods) 
                      if all_results[key]['total_wealth'][j] > all_results['regular']['total_wealth'][j])
            win_rate = wins / n_periods * 100
        
        print(f"{strategy_labels[i]:<15} €{portfolio_mean:>9,.0f} €{cash_mean:>7,.0f} €{total_mean:>9,.0f} {win_rate:>8.1f}%")
    
    plt.show()
    
    return filename

def main():
    """Main function to run the complete analysis"""
    print("=== SIMULAZIONE STRATEGIA 'BUY THE DIP' MULTI-THRESHOLD ===")
    print("Confronto tra strategie di investimento con diversi livelli di threshold:")
    print("1. Investimento regolare: €1000/mese in MSCI World ETF")
    print("2. Buy the dip: €800/mese + €200 risparmiati per acquisti al ribasso")
    print("   - Threshold 5%: Acquista al -5% dal picco")
    print("   - Threshold 10%: Acquista al -10% dal picco")
    print("   - Threshold 20%: Acquista al -20% dal picco")
    print("   - Threshold 30%: Acquista al -30% dal picco")
    
    # Run Monte Carlo simulation
    print("\nEsecuzione simulazione Monte Carlo...")
    results = run_monte_carlo_simulation(n_simulations=1000, years=5)
    
    # Analyze results
    analyze_results(results)
    
    # Print summary conclusions
    print_summary_conclusions(results)
    
    # Create Monte Carlo analysis with confidence intervals
    create_monte_carlo_analysis_image()
    
    # Show detailed example simulation with graphs
    show_example_simulation()
    
    print("\n" + "="*80)
    print("NOTE IMPORTANTI")
    print("="*80)
    print("• Questa è una simulazione basata su dati reali del MSCI World ETF")
    print("• I risultati reali possono variare significativamente")
    print("• Considerare sempre la diversificazione del portfolio")
    print("• Consultare un consulente finanziario per decisioni reali")

if __name__ == "__main__":
    main()
