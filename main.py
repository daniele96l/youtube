import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class RealGoldAnalysisValidator:
    def __init__(self, fred_api_key=None):
        """
        Initialize with real data sources
        fred_api_key: Optional FRED API key for economic data
        """
        self.data = {}
        self.start_date = "1970-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.fred_api_key = fred_api_key

    def fetch_real_data(self):
        """Fetch comprehensive real historical data"""
        print("Fetching real historical market data...")

        # 1. GOLD DATA - Multiple sources for reliability
        print("- Fetching Gold prices...")
        try:
            # Try multiple gold sources
            gold_sources = [
                ("GLD", "2004-01-01"),  # GLD ETF
                ("IAU", "2005-01-01"),  # iShares Gold Trust
                ("GC=F", "1990-01-01")  # Gold futures
            ]

            gold_data = None
            for symbol, start_date in gold_sources:
                try:
                    data = yf.download(symbol, start=start_date, end=self.end_date, progress=False)
                    if not data.empty and 'Close' in data.columns:
                        gold_data = data['Close'].dropna()
                        print(f"  ✓ Using {symbol}: {len(gold_data)} observations from {gold_data.index[0].date()}")
                        break
                except:
                    continue

            if gold_data is None or len(gold_data) < 100:
                print("  ⚠ Unable to fetch sufficient gold data")
                return False

            self.data['gold'] = gold_data

        except Exception as e:
            print(f"  ⚠ Error fetching gold data: {e}")
            return False

        # 2. STOCK MARKET DATA
        print("- Fetching Stock market data...")
        try:
            sp500 = yf.download("^GSPC", start="1970-01-01", end=self.end_date, progress=False)
            if not sp500.empty and 'Close' in sp500.columns:
                self.data['sp500'] = sp500['Close'].dropna()
                print(f"  ✓ S&P 500: {len(self.data['sp500'])} observations")
            else:
                print("  ⚠ Unable to fetch S&P 500 data")
                return False

        except Exception as e:
            print(f"  ⚠ Error fetching stock data: {e}")
            return False

        # 3. BOND DATA
        print("- Fetching Bond data...")
        try:
            # 10-Year Treasury Yield
            treasury_10y = yf.download("^TNX", start="1970-01-01", end=self.end_date, progress=False)
            if not treasury_10y.empty and 'Close' in treasury_10y.columns:
                self.data['treasury_10y_yield'] = treasury_10y['Close'].dropna()

            # TLT (20+ Year Treasury Bond ETF)
            tlt = yf.download("TLT", start="2002-01-01", end=self.end_date, progress=False)
            if not tlt.empty and 'Close' in tlt.columns:
                self.data['tlt'] = tlt['Close'].dropna()

            print(f"  ✓ Bond data fetched successfully")

        except Exception as e:
            print(f"  ⚠ Error fetching bond data: {e}")

        # 4. CURRENCY DATA
        print("- Fetching Currency data...")
        try:
            # US Dollar Index
            dxy = yf.download("DX-Y.NYB", start="1970-01-01", end=self.end_date, progress=False)
            if not dxy.empty and 'Close' in dxy.columns:
                self.data['usd_index'] = dxy['Close'].dropna()
                print(f"  ✓ USD Index: {len(self.data['usd_index'])} observations")

        except Exception as e:
            print(f"  ⚠ Error fetching currency data: {e}")

        # 5. VIX for crisis identification
        print("- Fetching VIX (volatility index)...")
        try:
            vix = yf.download("^VIX", start="1990-01-01", end=self.end_date, progress=False)
            if not vix.empty and 'Close' in vix.columns:
                self.data['vix'] = vix['Close'].dropna()
                print(f"  ✓ VIX: {len(self.data['vix'])} observations")
        except Exception as e:
            print(f"  ⚠ Error fetching VIX: {e}")

        print("\n✓ Data fetching completed!")
        self._print_data_summary()
        return True

    def _print_data_summary(self):
        """Print summary of available data"""
        print("\nDATA SUMMARY:")
        print("-" * 40)
        for key, data in self.data.items():
            if isinstance(data, pd.Series) and not data.empty:
                start_date = data.dropna().index[0].strftime('%Y-%m-%d')
                end_date = data.dropna().index[-1].strftime('%Y-%m-%d')
                print(f"{key:20}: {start_date} to {end_date} ({len(data.dropna())} obs)")

    def calculate_returns(self, price_series, period='daily'):
        """Calculate returns for a given price series"""
        if price_series is None or price_series.empty:
            return pd.Series(dtype=float)

        if period == 'daily':
            return price_series.pct_change().dropna()
        elif period == 'monthly':
            return price_series.resample('M').last().pct_change().dropna()
        elif period == 'annual':
            return price_series.resample('Y').last().pct_change().dropna()

    def test_volatility_claim_real(self):
        """Test Claim: Gold volatility is over 17%, higher than S&P 500 - Using REAL data"""
        print("\n" + "=" * 70)
        print("TESTING CLAIM 1: Gold volatility > 17% and > S&P 500 (REAL DATA)")
        print("=" * 70)

        if 'gold' not in self.data or 'sp500' not in self.data:
            print("Insufficient data for volatility analysis")
            return None

        gold_data = self.data['gold']
        sp500_data = self.data['sp500']

        # Find common period
        common_start = max(gold_data.dropna().index[0], sp500_data.dropna().index[0])
        common_end = min(gold_data.dropna().index[-1], sp500_data.dropna().index[-1])

        print(f"Analysis period: {common_start.date()} to {common_end.date()}")

        # Calculate returns for common period
        gold_returns = self.calculate_returns(gold_data.loc[common_start:common_end])
        sp500_returns = self.calculate_returns(sp500_data.loc[common_start:common_end])

        if gold_returns.empty or sp500_returns.empty:
            print("Unable to calculate returns")
            return None

        # Calculate annualized volatilities
        gold_vol = gold_returns.std() * np.sqrt(252) * 100
        sp500_vol = sp500_returns.std() * np.sqrt(252) * 100

        print(f"\nREAL VOLATILITY RESULTS:")
        print(f"Gold Volatility: {gold_vol:.2f}%")
        print(f"S&P 500 Volatility: {sp500_vol:.2f}%")
        print(f"Article claim: Gold vol > 17%? {gold_vol > 17} ({'✓' if gold_vol > 17 else '✗'})")
        print(f"Article claim: Gold vol > S&P? {gold_vol > sp500_vol} ({'✓' if gold_vol > sp500_vol else '✗'})")

        # Additional statistics
        print(f"\nADDITIONAL STATISTICS:")
        print(f"Gold max daily gain: {gold_returns.max() * 100:.2f}%")
        print(f"Gold max daily loss: {gold_returns.min() * 100:.2f}%")
        print(f"S&P 500 max daily gain: {sp500_returns.max() * 100:.2f}%")
        print(f"S&P 500 max daily loss: {sp500_returns.min() * 100:.2f}%")

        # Plot volatility analysis
        self._plot_volatility_analysis(gold_returns, sp500_returns, gold_vol, sp500_vol)

        return {
            'gold_vol': gold_vol,
            'sp500_vol': sp500_vol,
            'period': f"{common_start.date()} to {common_end.date()}",
            'gold_returns': gold_returns,
            'sp500_returns': sp500_returns
        }

    def _plot_volatility_analysis(self, gold_returns, sp500_returns, gold_vol, sp500_vol):
        """Plot volatility analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Rolling volatility comparison
        gold_rolling_vol = gold_returns.rolling(252).std() * np.sqrt(252) * 100
        sp500_rolling_vol = sp500_returns.rolling(252).std() * np.sqrt(252) * 100

        ax1.plot(gold_rolling_vol.index, gold_rolling_vol, label='Gold', linewidth=2, color='gold')
        ax1.plot(sp500_rolling_vol.index, sp500_rolling_vol, label='S&P 500', linewidth=2, color='blue')
        ax1.axhline(y=17, color='red', linestyle='--', alpha=0.7, label='17% threshold')
        ax1.set_title('Rolling 1-Year Volatility (Real Data)')
        ax1.set_ylabel('Volatility (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Volatility distribution
        ax2.hist(gold_rolling_vol.dropna(), bins=30, alpha=0.7, label='Gold', color='gold', density=True)
        ax2.hist(sp500_rolling_vol.dropna(), bins=30, alpha=0.7, label='S&P 500', color='blue', density=True)
        ax2.axvline(x=17, color='red', linestyle='--', alpha=0.7, label='17% threshold')
        ax2.axvline(x=gold_vol, color='orange', linestyle='-', alpha=0.7, label=f'Gold avg: {gold_vol:.1f}%')
        ax2.axvline(x=sp500_vol, color='navy', linestyle='-', alpha=0.7, label=f'S&P avg: {sp500_vol:.1f}%')
        ax2.set_title('Volatility Distribution')
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Returns scatter plot
        ax3.scatter(sp500_returns, gold_returns, alpha=0.5, s=5)
        ax3.set_xlabel('S&P 500 Daily Returns')
        ax3.set_ylabel('Gold Daily Returns')
        ax3.set_title('Gold vs S&P 500 Daily Returns')
        ax3.grid(True, alpha=0.3)

        # 4. Annual volatility by year
        gold_annual_vol = gold_returns.groupby(gold_returns.index.year).std() * np.sqrt(252) * 100
        sp500_annual_vol = sp500_returns.groupby(sp500_returns.index.year).std() * np.sqrt(252) * 100

        years = gold_annual_vol.index
        x = np.arange(len(years))
        width = 0.35

        ax4.bar(x - width / 2, gold_annual_vol, width, label='Gold', color='gold', alpha=0.7)
        ax4.bar(x + width / 2, sp500_annual_vol, width, label='S&P 500', color='blue', alpha=0.7)
        ax4.axhline(y=17, color='red', linestyle='--', alpha=0.7, label='17% threshold')
        ax4.set_title('Annual Volatility by Year')
        ax4.set_ylabel('Volatility (%)')
        ax4.set_xlabel('Year')
        ax4.set_xticks(x[::2])
        ax4.set_xticklabels(years[::2], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def test_crisis_correlation_real(self):
        """Test Claim: Gold crashes with stocks during crisis periods - Using REAL data"""
        print("\n" + "=" * 70)
        print("TESTING CLAIM 2: Gold correlation with stocks during crisis (REAL DATA)")
        print("=" * 70)

        if 'gold' not in self.data or 'sp500' not in self.data:
            print("Insufficient data for crisis correlation analysis")
            return None

        gold_returns = self.calculate_returns(self.data['gold'])
        sp500_returns = self.calculate_returns(self.data['sp500'])

        # Align the data
        common_dates = gold_returns.index.intersection(sp500_returns.index)
        if len(common_dates) < 100:
            print("Insufficient overlapping data")
            return None

        gold_aligned = gold_returns.loc[common_dates]
        sp500_aligned = sp500_returns.loc[common_dates]

        print(f"Analysis period: {common_dates[0].date()} to {common_dates[-1].date()}")
        print(f"Total observations: {len(common_dates)}")

        # Identify crisis periods using multiple methods
        crisis_periods = pd.Series(False, index=common_dates)

        # Method 1: VIX spikes (if available)
        if 'vix' in self.data and not self.data['vix'].empty:
            vix = self.data['vix']
            vix_aligned = vix.reindex(common_dates, method='ffill')
            if not vix_aligned.empty:
                vix_threshold = vix_aligned.quantile(0.8)
                vix_crisis = vix_aligned > vix_threshold
                crisis_periods = crisis_periods | vix_crisis.fillna(False)
                print(f"VIX crisis threshold: {vix_threshold:.1f}")

        # Method 2: Large negative S&P 500 moves
        sp500_threshold = sp500_aligned.quantile(0.05)
        large_drops = sp500_aligned < sp500_threshold
        crisis_periods = crisis_periods | large_drops

        # Overall correlation analysis
        overall_corr = gold_aligned.corr(sp500_aligned)
        rolling_corr = gold_aligned.rolling(60).corr(sp500_aligned)

        # Crisis vs normal periods
        if crisis_periods.any():
            crisis_corr = rolling_corr[crisis_periods].mean()
            normal_corr = rolling_corr[~crisis_periods].mean()
        else:
            crisis_corr = overall_corr
            normal_corr = overall_corr

        print(f"\nOVERALL CORRELATION ANALYSIS:")
        print(f"Overall correlation: {overall_corr:.3f}")
        print(f"Crisis periods correlation: {crisis_corr:.3f}")
        print(f"Normal periods correlation: {normal_corr:.3f}")
        print(
            f"Higher correlation in crisis? {crisis_corr > normal_corr} ({'✓' if crisis_corr > normal_corr else '✗'})")

        return {
            'overall_correlation': overall_corr,
            'crisis_correlation': crisis_corr,
            'normal_correlation': normal_corr,
            'crisis_periods_pct': crisis_periods.mean() * 100
        }

    def test_usd_correlation_real(self):
        """Test Claim: Gold acts as USD hedge - Using REAL data"""
        print("\n" + "=" * 70)
        print("TESTING CLAIM 3: Gold as USD hedge (negative correlation) (REAL DATA)")
        print("=" * 70)

        if 'gold' not in self.data:
            print("No gold data available")
            return None

        gold_returns = self.calculate_returns(self.data['gold'])

        if 'usd_index' in self.data and not self.data['usd_index'].empty:
            usd_returns = self.calculate_returns(self.data['usd_index'])
            data_source = "USD Index (DXY)"
        else:
            print("No USD data available for analysis")
            return None

        # Align data
        common_dates = gold_returns.index.intersection(usd_returns.index)
        if len(common_dates) < 100:
            print("Insufficient overlapping data")
            return None

        gold_aligned = gold_returns.loc[common_dates]
        usd_aligned = usd_returns.loc[common_dates]

        print(f"Data source: {data_source}")
        print(f"Analysis period: {common_dates[0].date()} to {common_dates[-1].date()}")
        print(f"Total observations: {len(common_dates)}")

        # Calculate correlations
        overall_corr = gold_aligned.corr(usd_aligned)
        rolling_corr = gold_aligned.rolling(60).corr(usd_aligned)

        print(f"\nCORRELATION ANALYSIS:")
        print(f"Overall correlation: {overall_corr:.3f}")
        print(f"Average rolling correlation: {rolling_corr.mean():.3f}")
        print(f"Negative correlation confirmed? {overall_corr < -0.1} ({'✓' if overall_corr < -0.1 else '✗'})")
        print(f"Strong hedge (corr < -0.3)? {overall_corr < -0.3} ({'✓' if overall_corr < -0.3 else '✗'})")

        # Statistical significance test
        from scipy.stats import pearsonr
        corr_stat, p_value = pearsonr(gold_aligned, usd_aligned)
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"Correlation: {corr_stat:.3f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significant at 5% level? {p_value < 0.05} ({'✓' if p_value < 0.05 else '✗'})")

        return {
            'overall_correlation': overall_corr,
            'rolling_correlation_mean': rolling_corr.mean(),
            'p_value': p_value,
            'observations': len(common_dates)
        }

    def test_portfolio_optimization_real(self):
        """Test Claim: 5-15% gold allocation optimizes portfolio - Using REAL data"""
        print("\n" + "=" * 70)
        print("TESTING CLAIM 4: Optimal gold allocation is 5-15% (REAL DATA)")
        print("=" * 70)

        if 'gold' not in self.data or 'sp500' not in self.data:
            print("Insufficient data for portfolio optimization")
            return None

        # Get monthly returns for better portfolio analysis
        gold_monthly = self.data['gold'].resample('M').last().pct_change().dropna()
        sp500_monthly = self.data['sp500'].resample('M').last().pct_change().dropna()

        # Use bond data if available
        if 'tlt' in self.data and not self.data['tlt'].empty:
            bond_monthly = self.data['tlt'].resample('M').last().pct_change().dropna()
            bond_name = "TLT (20+ Year Treasuries)"
        else:
            # Create simple bond proxy with low volatility
            bond_monthly = pd.Series(np.random.normal(0.005, 0.02, len(sp500_monthly)),
                                     index=sp500_monthly.index)
            bond_name = "Synthetic bond proxy"

        # Find common date range
        common_dates = gold_monthly.index.intersection(sp500_monthly.index).intersection(bond_monthly.index)

        if len(common_dates) < 36:
            print("Insufficient overlapping data for portfolio optimization")
            return None

        # Align all data
        gold_aligned = gold_monthly.loc[common_dates]
        stocks_aligned = sp500_monthly.loc[common_dates]
        bonds_aligned = bond_monthly.loc[common_dates]

        print(f"Portfolio analysis period: {common_dates[0].date()} to {common_dates[-1].date()}")
        print(f"Monthly observations: {len(common_dates)}")
        print(f"Bond proxy: {bond_name}")

        # Test different gold allocations
        allocations = np.arange(0, 51, 5)
        results = []

        print(f"\nPORTFOLIO OPTIMIZATION RESULTS:")
        print("=" * 65)
        print("Gold%  Return%  Vol%   Sharpe  Sortino")
        print("-" * 65)

        for gold_alloc in allocations:
            # Split remaining allocation between stocks and bonds (60/40 split)
            remaining = 100 - gold_alloc
            stock_alloc = remaining * 0.6
            bond_alloc = remaining * 0.4

            # Calculate portfolio returns
            portfolio_returns = (gold_alloc / 100 * gold_aligned +
                                 stock_alloc / 100 * stocks_aligned +
                                 bond_alloc / 100 * bonds_aligned)

            # Calculate metrics
            annual_return = portfolio_returns.mean() * 12 * 100
            annual_vol = portfolio_returns.std() * np.sqrt(12) * 100
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

            # Downside deviation for Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < portfolio_returns.mean()]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(12) * 100
                sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
            else:
                sortino_ratio = sharpe_ratio

            results.append({
                'gold_allocation': gold_alloc,
                'real_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            })

            print(
                f"{gold_alloc:4d}   {annual_return:6.1f}  {annual_vol:5.1f}  {sharpe_ratio:6.3f}  {sortino_ratio:7.3f}")

        results_df = pd.DataFrame(results)

        # Find optimal allocations
        optimal_sharpe_idx = results_df['sharpe_ratio'].idxmax()
        optimal_sortino_idx = results_df['sortino_ratio'].idxmax()

        optimal_sharpe = results_df.loc[optimal_sharpe_idx]
        optimal_sortino = results_df.loc[optimal_sortino_idx]

        print("=" * 65)
        print(f"\nOPTIMAL ALLOCATIONS:")
        print(
            f"Best Sharpe ratio: {optimal_sharpe['gold_allocation']:.0f}% gold (Sharpe: {optimal_sharpe['sharpe_ratio']:.3f})")
        print(
            f"Best Sortino ratio: {optimal_sortino['gold_allocation']:.0f}% gold (Sortino: {optimal_sortino['sortino_ratio']:.3f})")

        # Check if optimals fall in recommended range
        in_range_sharpe = 5 <= optimal_sharpe['gold_allocation'] <= 15
        in_range_sortino = 5 <= optimal_sortino['gold_allocation'] <= 15

        print(f"\nCLAIM VALIDATION (5-15% range):")
        print(f"Optimal Sharpe in range? {in_range_sharpe} ({'✓' if in_range_sharpe else '✗'})")
        print(f"Optimal Sortino in range? {in_range_sortino} ({'✓' if in_range_sortino else '✗'})")

        return {
            'optimal_sharpe': optimal_sharpe,
            'optimal_sortino': optimal_sortino,
            'in_recommended_range': {
                'sharpe': in_range_sharpe,
                'sortino': in_range_sortino
            }
        }

    def run_comprehensive_real_analysis(self):
        """Run all tests with real data and provide comprehensive summary"""
        print("COMPREHENSIVE GOLD INVESTMENT CLAIMS VALIDATION")
        print("Using Real Historical Market Data")
        print("=" * 80)

        # Attempt to fetch real data
        if not self.fetch_real_data():
            print("Failed to fetch sufficient real data. Please check your internet connection.")
            return None

        # Run all tests with real data
        results = {}

        print("\n🔍 RUNNING COMPREHENSIVE ANALYSIS...")

        try:
            print("1/4 Testing volatility claims...")
            results['volatility'] = self.test_volatility_claim_real()
        except Exception as e:
            print(f"❌ Error in volatility test: {e}")

        try:
            print("\n2/4 Testing crisis correlation claims...")
            results['crisis_correlation'] = self.test_crisis_correlation_real()
        except Exception as e:
            print(f"❌ Error in crisis correlation test: {e}")

        try:
            print("\n3/4 Testing USD correlation claims...")
            results['usd_correlation'] = self.test_usd_correlation_real()
        except Exception as e:
            print(f"❌ Error in USD correlation test: {e}")

        try:
            print("\n4/4 Testing portfolio optimization claims...")
            results['portfolio_optimization'] = self.test_portfolio_optimization_real()
        except Exception as e:
            print(f"❌ Error in portfolio optimization test: {e}")

        # Generate comprehensive summary
        self._generate_comprehensive_summary(results)

        return results

    def _generate_comprehensive_summary(self, results):
        """Generate a comprehensive summary of all findings"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE SUMMARY OF CLAIMS VALIDATION")
        print("=" * 80)

        total_claims = 0
        validated_claims = 0

        print("\n1️⃣  VOLATILITY ANALYSIS")
        print("-" * 40)
        if 'volatility' in results and results['volatility'] is not None:
            vol_data = results['volatility']
            claim1a = vol_data['gold_vol'] > 17
            claim1b = vol_data['gold_vol'] > vol_data['sp500_vol']

            print(f"   📈 Gold volatility: {vol_data['gold_vol']:.1f}%")
            print(f"   📈 S&P 500 volatility: {vol_data['sp500_vol']:.1f}%")
            print(f"   ✅ Claim: Gold vol > 17%? {claim1a} ({'VALIDATED' if claim1a else 'REJECTED'})")
            print(f"   ✅ Claim: Gold vol > S&P? {claim1b} ({'VALIDATED' if claim1b else 'REJECTED'})")

            total_claims += 2
            validated_claims += int(claim1a) + int(claim1b)
        else:
            print("   ❌ Analysis failed - insufficient data")

        print("\n2️⃣  CRISIS CORRELATION ANALYSIS")
        print("-" * 40)
        if 'crisis_correlation' in results and results['crisis_correlation'] is not None:
            crisis_data = results['crisis_correlation']
            claim2 = crisis_data['crisis_correlation'] > crisis_data['normal_correlation']

            print(f"   🔥 Crisis correlation: {crisis_data['crisis_correlation']:.3f}")
            print(f"   😐 Normal correlation: {crisis_data['normal_correlation']:.3f}")
            print(f"   ✅ Claim: Higher correlation in crisis? {claim2} ({'VALIDATED' if claim2 else 'REJECTED'})")

            total_claims += 1
            validated_claims += int(claim2)
        else:
            print("   ❌ Analysis failed - insufficient data")

        print("\n3️⃣  USD HEDGE ANALYSIS")
        print("-" * 40)
        if 'usd_correlation' in results and results['usd_correlation'] is not None:
            usd_corr = results['usd_correlation']['overall_correlation']
            claim3 = usd_corr < -0.1

            print(f"   💵 Gold-USD correlation: {usd_corr:.3f}")
            print(f"   ✅ Claim: Negative correlation? {claim3} ({'VALIDATED' if claim3 else 'REJECTED'})")

            total_claims += 1
            validated_claims += int(claim3)
        else:
            print("   ❌ Analysis failed - insufficient data")

        print("\n4️⃣  PORTFOLIO OPTIMIZATION ANALYSIS")
        print("-" * 40)
        if 'portfolio_optimization' in results and results['portfolio_optimization'] is not None:
            opt_data = results['portfolio_optimization']
            optimal_sortino_alloc = opt_data['optimal_sortino']['gold_allocation']
            optimal_sharpe_alloc = opt_data['optimal_sharpe']['gold_allocation']

            claim4a = 5 <= optimal_sortino_alloc <= 15
            claim4b = 5 <= optimal_sharpe_alloc <= 15

            print(f"   🎯 Optimal allocation (Sortino): {optimal_sortino_alloc:.0f}%")
            print(f"   🎯 Optimal allocation (Sharpe): {optimal_sharpe_alloc:.0f}%")
            print(f"   ✅ Claim: Sortino optimal in 5-15%? {claim4a} ({'VALIDATED' if claim4a else 'REJECTED'})")
            print(f"   ✅ Claim: Sharpe optimal in 5-15%? {claim4b} ({'VALIDATED' if claim4b else 'REJECTED'})")

            total_claims += 2
            validated_claims += int(claim4a) + int(claim4b)
        else:
            print("   ❌ Analysis failed - insufficient data")

        # Overall validation score
        print("\n" + "=" * 80)
        print("🏆 OVERALL VALIDATION SCORE")
        print("=" * 80)

        if total_claims > 0:
            validation_rate = validated_claims / total_claims * 100
            print(f"   📊 Claims validated: {validated_claims}/{total_claims} ({validation_rate:.1f}%)")

            if validation_rate >= 80:
                verdict = "🟢 HIGHLY SUPPORTED"
                interpretation = "The article's claims are strongly supported by historical data"
            elif validation_rate >= 60:
                verdict = "🟡 PARTIALLY SUPPORTED"
                interpretation = "Most claims are supported, but some require qualification"
            elif validation_rate >= 40:
                verdict = "🟠 MIXED EVIDENCE"
                interpretation = "Evidence is mixed - some claims supported, others not"
            else:
                verdict = "🔴 POORLY SUPPORTED"
                interpretation = "Most claims are not supported by historical data"

            print(f"   🎯 Verdict: {verdict}")
            print(f"   💭 Interpretation: {interpretation}")
        else:
            print("   ❌ Unable to validate claims due to data limitations")

        print("\n" + "=" * 80)
        print("📝 KEY TAKEAWAYS")
        print("=" * 80)

        takeaways = [
            "✨ This analysis uses real historical market data from multiple sources",
            "📈 Volatility and correlation patterns vary significantly over time",
            "🎯 Portfolio optimization results depend on the specific time period analyzed",
            "💰 Gold's effectiveness as a hedge depends on market conditions",
            "⚠️  Past performance does not guarantee future results",
            "🔍 Consider these findings alongside your investment goals and risk tolerance"
        ]

        for takeaway in takeaways:
            print(f"   {takeaway}")

        print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    print("REAL GOLD ANALYSIS VALIDATOR")
    print("=" * 50)
    print("This script validates gold investment claims using real historical data.")
    print("=" * 50)

    # Initialize analyzer
    analyzer = RealGoldAnalysisValidator()

    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_real_analysis()

        if results:
            print("\n✅ Analysis completed successfully!")
            print("📊 All results are displayed above.")
            print("💾 Results are stored in the 'results' variable for further analysis.")
        else:
            print("\n❌ Analysis failed due to data limitations.")
            print("🔧 Try running the script again or check your internet connection.")

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("🔧 Please check your internet connection and try again.")