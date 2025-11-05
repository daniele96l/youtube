import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class ETFPortfolioAnalyzer:
    def __init__(self, swda_path, sxr7_path):
        self.swda_path = swda_path
        self.sxr7_path = sxr7_path
        self.load_data()

    def load_data(self):
        print("\nLoading data...")
        self.swda_df = pd.read_csv(self.swda_path)
        self.swda_df['Date'] = pd.to_datetime(self.swda_df['Date'], format='%m/%Y')
        self.swda_df.columns = ['Date', 'SWDA']
        self.swda_df.set_index('Date', inplace=True)
        print(f"   SWDA: {len(self.swda_df)} months")

        self.sxr7_df = pd.read_csv(self.sxr7_path)
        self.sxr7_df['Date'] = pd.to_datetime(self.sxr7_df['Date'], format='%m/%Y')
        self.sxr7_df.columns = ['Date', 'SXR7']
        self.sxr7_df.set_index('Date', inplace=True)
        print(f"   SXR7: {len(self.sxr7_df)} months")

        self.combined_df = pd.merge(self.swda_df, self.sxr7_df,
                                     left_index=True, right_index=True, how='inner')
        self.returns_df = self.combined_df.pct_change().dropna()

        print(f"Range: {self.combined_df.index.min().strftime('%Y-%m')} to {self.combined_df.index.max().strftime('%Y-%m')}")
        print(f"   Total: {len(self.combined_df)} months\n")

    def create_portfolio_allocations(self, step=2.5):
        allocations = []
        for swda_weight in np.arange(0, 100 + step, step):
            sxr7_weight = 100 - swda_weight
            allocations.append((swda_weight / 100, sxr7_weight / 100))
        return allocations

    def calculate_portfolio_returns(self, swda_weight, sxr7_weight):
        return (self.returns_df['SWDA'] * swda_weight +
                self.returns_df['SXR7'] * sxr7_weight)

    def calculate_max_drawdown(self, simulations):
        max_drawdowns = []
        for sim in simulations:
            cumulative = sim
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdowns.append(drawdown.min())
        return np.mean(max_drawdowns)

    def monte_carlo_simulation(self, swda_weight, sxr7_weight, years, n_simulations=10000):
        portfolio_returns = self.calculate_portfolio_returns(swda_weight, sxr7_weight)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        n_months = years * 12

        simulations = np.zeros((n_simulations, n_months))
        monthly_returns_sim = np.zeros((n_simulations, n_months))
        for i in range(n_simulations):
            monthly_returns = np.random.normal(mean_return, std_return, n_months)
            monthly_returns_sim[i] = monthly_returns
            simulations[i] = np.cumprod(1 + monthly_returns)

        final_values = simulations[:, -1]
        cagr = (final_values ** (1/years) - 1) * 100

        # Calculate annualized volatility: std of monthly returns * sqrt(12) for each simulation, then average
        # Vectorized calculation for better performance
        monthly_stds = np.std(monthly_returns_sim, axis=1)
        annualized_volatilities = monthly_stds * np.sqrt(12) * 100
        volatility = np.mean(annualized_volatilities)
        
        # Annualized mean return for this time horizon (in %)
        annualized_mean_return = np.mean(cagr)
        
        # Sharpe ratio: (Mean Return - Risk Free) / Volatility (assuming 0% risk-free rate)
        # Both are in %, so division gives ratio directly
        sharpe = annualized_mean_return / volatility if volatility > 0 else 0

        # Sortino ratio: use downside deviation (annualized) - std of negative monthly returns
        downside_volatilities = []
        for i in range(n_simulations):
            monthly_returns_i = monthly_returns_sim[i]
            downside_months = monthly_returns_i[monthly_returns_i < 0]
            if len(downside_months) > 0:
                downside_vol = np.std(downside_months) * np.sqrt(12) * 100
                downside_volatilities.append(downside_vol)
        
        downside_volatility = np.mean(downside_volatilities) if len(downside_volatilities) > 0 else volatility
        sortino = annualized_mean_return / downside_volatility if downside_volatility > 0 else 0

        max_dd = self.calculate_max_drawdown(simulations)

        return {
            'cagr_mean': np.mean(cagr),
            'cagr_median': np.median(cagr),
            'cagr_std': np.std(cagr),
            'cagr_5th': np.percentile(cagr, 5),
            'cagr_95th': np.percentile(cagr, 95),
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd
        }

    def run_full_analysis(self, time_horizons=[2, 5, 7, 10, 15, 20], n_simulations=10000):
        allocations = self.create_portfolio_allocations()
        results_list = []
        total_scenarios = len(allocations) * len(time_horizons)
        current_scenario = 0

        print(f"{'='*80}")
        print(f"Running Monte Carlo Simulations")
        print(f"{'='*80}")
        print(f"Total scenarios: {total_scenarios}")
        print(f"Simulations per scenario: {n_simulations:,}\n")

        for swda_weight, sxr7_weight in allocations:
            for years in time_horizons:
                current_scenario += 1
                if current_scenario % 25 == 0 or current_scenario == total_scenarios:
                    print(f"Progress: {current_scenario}/{total_scenarios}")

                results = self.monte_carlo_simulation(swda_weight, sxr7_weight, years, n_simulations)

                results_list.append({
                    'SWDA_Weight': swda_weight * 100,
                    'SXR7_Weight': sxr7_weight * 100,
                    'Years': years,
                    'CAGR_Mean (%)': results['cagr_mean'],
                    'CAGR_Median (%)': results['cagr_median'],
                    'CAGR_StdDev (%)': results['cagr_std'],
                    'CAGR_5th_Percentile (%)': results['cagr_5th'],
                    'CAGR_95th_Percentile (%)': results['cagr_95th'],
                    'Volatility (%)': results['volatility'],
                    'Sharpe_Ratio': results['sharpe'],
                    'Sortino_Ratio': results['sortino'],
                    'Max_Drawdown (%)': results['max_drawdown']
                })

        print(f"\nAll scenarios completed!\n")
        self.results_df = pd.DataFrame(results_list)
        return self.results_df

    def plot_metric_evolution(self, metric='Sharpe_Ratio'):
        fig = plt.figure(figsize=(12, 8))

        time_horizons = sorted(self.results_df['Years'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_horizons)))

        lower_is_better = metric in ['Volatility (%)', 'Max_Drawdown (%)']

        ax1 = plt.subplot(1, 1, 1)

        best_allocations = []
        for idx, years in enumerate(time_horizons):
            data = self.results_df[self.results_df['Years'] == years]
            ax1.plot(data['SWDA_Weight'], data[metric],
                    marker='o', linewidth=2.5, markersize=5,
                    color=colors[idx], label=f'{years}Y', alpha=0.8)

            if metric == 'Max_Drawdown (%)':
                best_idx = data[metric].idxmax()
            elif lower_is_better:
                best_idx = data[metric].idxmin()
            else:
                best_idx = data[metric].idxmax()

            best_alloc = data.loc[best_idx, 'SWDA_Weight']
            best_value = data.loc[best_idx, metric]
            best_allocations.append((years, best_alloc, best_value))

            ax1.scatter(best_alloc, best_value, s=400, marker='*',
                       color='red', edgecolors='yellow', linewidths=2, zorder=10)

        ax1.set_xlabel('SWDA Weight (%)', fontsize=13, fontweight='bold')
        ax1.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax1.set_title(f'{metric} vs SWDA Allocation (* = Best)',
                     fontsize=15, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=10, ncol=3)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\n{'='*80}")
        print(f"BEST ALLOCATIONS FOR {metric}")
        print(f"{'='*80}")
        for years, alloc, value in best_allocations:
            print(f"   {years:2d} Years: {alloc:5.1f}% SWDA -> {metric} = {value:8.3f}")
        print(f"{'='*80}\n")

    def plot_all_metrics_comparison(self):
        metrics = ['CAGR_Mean (%)', 'Volatility (%)', 'Sharpe_Ratio', 'Sortino_Ratio']
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        time_horizons = sorted(self.results_df['Years'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_horizons)))

        for idx, metric in enumerate(metrics):
            lower_is_better = metric in ['Volatility (%)']

            for t_idx, years in enumerate(time_horizons):
                data = self.results_df[self.results_df['Years'] == years]
                axes[idx].plot(data['SWDA_Weight'], data[metric],
                             marker='o', linewidth=2, markersize=4,
                             color=colors[t_idx], label=f'{years}Y', alpha=0.8)

                if lower_is_better:
                    best_idx = data[metric].idxmin()
                else:
                    best_idx = data[metric].idxmax()

                best_alloc = data.loc[best_idx, 'SWDA_Weight']
                best_value = data.loc[best_idx, metric]

                axes[idx].scatter(best_alloc, best_value, s=250, marker='*',
                                color='red', edgecolors='yellow', linewidths=1.5, zorder=10)

            axes[idx].set_xlabel('SWDA Weight (%)', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{metric} Evolution (* = Best)', fontsize=13, fontweight='bold')
            axes[idx].legend(loc='best', fontsize=8, ncol=3)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_max_drawdown_analysis(self):
        fig = plt.figure(figsize=(18, 10))

        time_horizons = sorted(self.results_df['Years'].unique())
        colors = plt.cm.plasma(np.linspace(0, 1, len(time_horizons)))

        ax1 = plt.subplot(2, 2, (1, 2))

        best_allocations = []
        for idx, years in enumerate(time_horizons):
            data = self.results_df[self.results_df['Years'] == years]
            ax1.plot(data['SWDA_Weight'], data['Max_Drawdown (%)'],
                    marker='o', linewidth=2.5, markersize=5,
                    color=colors[idx], label=f'{years}Y', alpha=0.8)

            best_idx = data['Max_Drawdown (%)'].idxmax()
            best_alloc = data.loc[best_idx, 'SWDA_Weight']
            best_value = data.loc[best_idx, 'Max_Drawdown (%)']
            best_allocations.append((years, best_alloc, best_value))

            ax1.scatter(best_alloc, best_value, s=400, marker='*',
                       color='red', edgecolors='yellow', linewidths=2, zorder=10)

        ax1.set_xlabel('SWDA Weight (%)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Max Drawdown (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Max Drawdown vs SWDA Allocation (* = Lowest Drawdown)',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=10, ncol=3)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax3 = plt.subplot(2, 2, 3)
        for idx, years in enumerate(time_horizons):
            data = self.results_df[self.results_df['Years'] == years]
            scatter = ax3.scatter(data['Max_Drawdown (%)'], data['CAGR_Mean (%)'],
                                 c=data['SWDA_Weight'], cmap='viridis',
                                 s=80, alpha=0.6, label=f'{years}Y')

        ax3.set_xlabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('CAGR Mean (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Risk-Return: Drawdown vs CAGR', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('SWDA %', fontsize=10)

        ax4 = plt.subplot(2, 2, 4)
        for idx, years in enumerate(time_horizons):
            data = self.results_df[self.results_df['Years'] == years]
            ax4.scatter(data['Volatility (%)'], data['Max_Drawdown (%)'],
                       color=colors[idx], s=80, alpha=0.6, label=f'{years}Y')

        ax4.set_xlabel('Volatility (%)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Volatility vs Max Drawdown', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\n{'='*80}")
        print(f"BEST ALLOCATIONS FOR MINIMUM DRAWDOWN")
        print(f"{'='*80}")
        for years, alloc, value in best_allocations:
            print(f"   {years:2d} Years: {alloc:5.1f}% SWDA -> Max Drawdown = {value:7.2f}%")
        print(f"{'='*80}\n")

    def plot_optimal_allocation_by_horizon(self):
        optimal = []
        for years in sorted(self.results_df['Years'].unique()):
            data = self.results_df[self.results_df['Years'] == years]
            optimal.append({
                'Years': years,
                'Max_Sharpe_SWDA%': data.loc[data['Sharpe_Ratio'].idxmax(), 'SWDA_Weight'],
                'Max_Sharpe': data['Sharpe_Ratio'].max(),
                'Max_Sortino_SWDA%': data.loc[data['Sortino_Ratio'].idxmax(), 'SWDA_Weight'],
                'Max_Sortino': data['Sortino_Ratio'].max(),
                'Max_CAGR_SWDA%': data.loc[data['CAGR_Mean (%)'].idxmax(), 'SWDA_Weight'],
                'Max_CAGR': data['CAGR_Mean (%)'].max(),
                'Min_Vol_SWDA%': data.loc[data['Volatility (%)'].idxmin(), 'SWDA_Weight'],
                'Min_Vol': data['Volatility (%)'].min(),
                'Min_DD_SWDA%': data.loc[data['Max_Drawdown (%)'].idxmax(), 'SWDA_Weight'],
                'Min_DD': data['Max_Drawdown (%)'].max()
            })

        optimal_df = pd.DataFrame(optimal)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        ax.plot(optimal_df['Years'], optimal_df['Max_Sharpe_SWDA%'],
                marker='o', linewidth=2.5, markersize=9, label='Max Sharpe', color='steelblue')
        ax.plot(optimal_df['Years'], optimal_df['Max_Sortino_SWDA%'],
                marker='s', linewidth=2.5, markersize=9, label='Max Sortino', color='darkgreen')
        ax.plot(optimal_df['Years'], optimal_df['Max_CAGR_SWDA%'],
                marker='^', linewidth=2.5, markersize=9, label='Max CAGR', color='darkred')
        ax.plot(optimal_df['Years'], optimal_df['Min_Vol_SWDA%'],
                marker='D', linewidth=2.5, markersize=9, label='Min Volatility', color='orange')
        ax.plot(optimal_df['Years'], optimal_df['Min_DD_SWDA%'],
                marker='*', linewidth=2.5, markersize=12, label='Min Drawdown', color='purple')

        ax.set_xlabel('Time Horizon (Years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimal SWDA Weight (%)', fontsize=12, fontweight='bold')
        ax.set_title('Optimal SWDA Allocation by Criterion', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()
        plt.show()

        return optimal_df

    def create_interactive_table(self):
        print("\n" + "="*100)
        print("RESULTS BY TIME HORIZON")
        print("="*100 + "\n")

        for years in sorted(self.results_df['Years'].unique()):
            data = self.results_df[self.results_df['Years'] == years]

            print(f"\n{'─'*100}")
            print(f"TIME HORIZON: {years} YEARS")
            print(f"{'─'*100}")

            key_allocations = data[data['SWDA_Weight'].isin([0, 25, 50, 75, 100])]
            display(key_allocations[['SWDA_Weight', 'SXR7_Weight', 'CAGR_Mean (%)',
                                     'Volatility (%)', 'Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown (%)']])

            max_sharpe_idx = data['Sharpe_Ratio'].idxmax()
            min_dd_idx = data['Max_Drawdown (%)'].idxmax()

            print(f"\nOPTIMAL (Max Sharpe): {data.loc[max_sharpe_idx, 'SWDA_Weight']:.1f}% SWDA")
            print(f"   Sharpe: {data.loc[max_sharpe_idx, 'Sharpe_Ratio']:.3f} | "
                  f"Sortino: {data.loc[max_sharpe_idx, 'Sortino_Ratio']:.3f} | "
                  f"CAGR: {data.loc[max_sharpe_idx, 'CAGR_Mean (%)']:.2f}% | "
                  f"Max DD: {data.loc[max_sharpe_idx, 'Max_Drawdown (%)']:.2f}%")

            print(f"\nOPTIMAL (Min Drawdown): {data.loc[min_dd_idx, 'SWDA_Weight']:.1f}% SWDA")
            print(f"   Max DD: {data.loc[min_dd_idx, 'Max_Drawdown (%)']:.2f}% | "
                  f"Sharpe: {data.loc[min_dd_idx, 'Sharpe_Ratio']:.3f} | "
                  f"CAGR: {data.loc[min_dd_idx, 'CAGR_Mean (%)']:.2f}%")

print("Class defined!")


analyzer = ETFPortfolioAnalyzer(
    swda_path='/Users/danieleligato/PycharmProjects/pythonProject4/iShares Core MSCI World UCITS ETF USD (Acc).csv',
    sxr7_path='/Users/danieleligato/PycharmProjects/pythonProject4/iShares Core MSCI EMU UCITS ETF EUR (Acc).csv'
)

results = analyzer.run_full_analysis(
    time_horizons=[5, 7, 10, 15, 20],
    n_simulations=10000
)


print("\n" + "="*100)
print("ALL METRICS COMPARISON (* = Best Allocation)")
print("="*100 + "\n")
analyzer.plot_all_metrics_comparison()


print("\n" + "="*100)
print("SHARPE RATIO ANALYSIS")
print("="*100)
analyzer.plot_metric_evolution('Sharpe_Ratio')

print("\n" + "="*100)
print("SORTINO RATIO ANALYSIS")
print("="*100)
analyzer.plot_metric_evolution('Sortino_Ratio')

print("\n" + "="*100)
print("CAGR ANALYSIS")
print("="*100)
analyzer.plot_metric_evolution('CAGR_Mean (%)')

print("\n" + "="*100)
print("VOLATILITY ANALYSIS")
print("="*100)
analyzer.plot_metric_evolution('Volatility (%)')


print("\n" + "="*100)
print("MAX DRAWDOWN COMPREHENSIVE ANALYSIS")
print("="*100 + "\n")
analyzer.plot_max_drawdown_analysis()


print("\n" + "="*100)
print("OPTIMAL ALLOCATIONS BY TIME HORIZON (All Criteria)")
print("="*100 + "\n")
optimal_df = analyzer.plot_optimal_allocation_by_horizon()
display(optimal_df)


analyzer.create_interactive_table()


print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100 + "\n")

summary = results.groupby('Years').agg({
    'CAGR_Mean (%)': ['min', 'max', 'mean'],
    'Volatility (%)': ['min', 'max', 'mean'],
    'Sharpe_Ratio': ['min', 'max', 'mean'],
    'Sortino_Ratio': ['min', 'max', 'mean'],
    'Max_Drawdown (%)': ['min', 'max', 'mean']
}).round(3)

display(summary)

print("\nANALYSIS COMPLETE!")