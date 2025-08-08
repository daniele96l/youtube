import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configurazione stile grafici
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CostOfLivingAnalysis:
    def __init__(self):
        """
        Classe per l'analisi comparativa del costo della vita
        Italia vs Vietnam con dati da multiple API
        """
        self.countries = {
            'Italy': {'code': 'IT', 'currency': 'EUR', 'city': 'Rome'},
            'Vietnam': {'code': 'VN', 'currency': 'VND', 'city': 'Ho Chi Minh City'}
        }

        # Dati di esempio realistici (da sostituire con API calls)
        self.cost_of_living_data = {
            'Italy': {
                'rent_1br_city_center': 900,
                'rent_1br_outside_center': 650,
                'meal_restaurant': 35,
                'meal_inexpensive': 15,
                'cappuccino': 1.5,
                'milk_1l': 1.2,
                'bread_500g': 1.8,
                'rice_1kg': 1.5,
                'eggs_12': 3.2,
                'chicken_1kg': 8.5,
                'beef_1kg': 18.0,
                'utilities_basic': 150,
                'internet': 28,
                'transport_pass': 35,
                'gasoline_1l': 1.65,
                'fitness_club': 50,
                'cinema_ticket': 8.5
            },
            'Vietnam': {
                'rent_1br_city_center': 400,
                'rent_1br_outside_center': 280,
                'meal_restaurant': 8,
                'meal_inexpensive': 3,
                'cappuccino': 1.2,
                'milk_1l': 1.1,
                'bread_500g': 0.8,
                'rice_1kg': 0.7,
                'eggs_12': 1.8,
                'chicken_1kg': 3.2,
                'beef_1kg': 9.5,
                'utilities_basic': 45,
                'internet': 8,
                'transport_pass': 12,
                'gasoline_1l': 0.85,
                'fitness_club': 25,
                'cinema_ticket': 3.5
            }
        }

        # Dati salari (in EUR equivalenti)
        self.salary_data = {
            'Italy': {
                'minimum_wage': 1200,
                'average_wage': 2500,
                'median_wage': 2100,
                'senior_developer': 4500,
                'teacher': 2200,
                'nurse': 2300,
                'engineer': 3800
            },
            'Vietnam': {
                'minimum_wage': 180,
                'average_wage': 450,
                'median_wage': 380,
                'senior_developer': 1200,
                'teacher': 350,
                'nurse': 400,
                'engineer': 800
            }
        }

        # Dati inflazione storica
        self.inflation_data = {
            'Italy': [0.8, 1.2, 1.9, 3.8, 5.7, 2.1],
            'Vietnam': [2.1, 2.8, 3.2, 4.1, 6.2, 3.5]
        }

        self.years = [2019, 2020, 2021, 2022, 2023, 2024]

    def get_exchange_rate(self):
        """Ottiene il tasso di cambio EUR/VND"""
        try:
            # API di esempio per tasso di cambio
            url = "https://api.exchangerate-api.com/v4/latest/EUR"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data['rates'].get('VND', 25000)  # Fallback se API non disponibile
            else:
                return 25000  # Tasso approssimativo EUR/VND
        except:
            return 25000

    def convert_vnd_to_eur(self, amount_vnd):
        """Converte da VND a EUR"""
        exchange_rate = self.get_exchange_rate()
        return amount_vnd / exchange_rate

    def create_cost_comparison_chart(self):
        """Crea grafici comparativi dei costi"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confronto Costo della Vita: Italia vs Vietnam', fontsize=16, fontweight='bold')

        # 1. Costi abitazione
        housing_costs = {
            'Centro città': [self.cost_of_living_data['Italy']['rent_1br_city_center'],
                             self.cost_of_living_data['Vietnam']['rent_1br_city_center']],
            'Fuori centro': [self.cost_of_living_data['Italy']['rent_1br_outside_center'],
                             self.cost_of_living_data['Vietnam']['rent_1br_outside_center']]
        }

        x = np.arange(len(housing_costs))
        width = 0.35

        axes[0, 0].bar(x - width / 2, [housing_costs['Centro città'][0], housing_costs['Fuori centro'][0]],
                       width, label='Italia', color='#2E86AB')
        axes[0, 0].bar(x + width / 2, [housing_costs['Centro città'][1], housing_costs['Fuori centro'][1]],
                       width, label='Vietnam', color='#A23B72')

        axes[0, 0].set_title('Costi Abitazione (€/mese)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(housing_costs.keys())
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Costi alimentari
        food_items = ['Ristorante', 'Pasto economico', 'Cappuccino', 'Latte 1L', 'Pollo 1kg']
        italy_food = [self.cost_of_living_data['Italy']['meal_restaurant'],
                      self.cost_of_living_data['Italy']['meal_inexpensive'],
                      self.cost_of_living_data['Italy']['cappuccino'],
                      self.cost_of_living_data['Italy']['milk_1l'],
                      self.cost_of_living_data['Italy']['chicken_1kg']]
        vietnam_food = [self.cost_of_living_data['Vietnam']['meal_restaurant'],
                        self.cost_of_living_data['Vietnam']['meal_inexpensive'],
                        self.cost_of_living_data['Vietnam']['cappuccino'],
                        self.cost_of_living_data['Vietnam']['milk_1l'],
                        self.cost_of_living_data['Vietnam']['chicken_1kg']]

        x = np.arange(len(food_items))
        axes[0, 1].bar(x - width / 2, italy_food, width, label='Italia', color='#2E86AB')
        axes[0, 1].bar(x + width / 2, vietnam_food, width, label='Vietnam', color='#A23B72')

        axes[0, 1].set_title('Costi Alimentari (€)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(food_items, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Servizi e trasporti
        services = ['Utilities', 'Internet', 'Trasporti', 'Palestra']
        italy_services = [self.cost_of_living_data['Italy']['utilities_basic'],
                          self.cost_of_living_data['Italy']['internet'],
                          self.cost_of_living_data['Italy']['transport_pass'],
                          self.cost_of_living_data['Italy']['fitness_club']]
        vietnam_services = [self.cost_of_living_data['Vietnam']['utilities_basic'],
                            self.cost_of_living_data['Vietnam']['internet'],
                            self.cost_of_living_data['Vietnam']['transport_pass'],
                            self.cost_of_living_data['Vietnam']['fitness_club']]

        x = np.arange(len(services))
        axes[1, 0].bar(x - width / 2, italy_services, width, label='Italia', color='#2E86AB')
        axes[1, 0].bar(x + width / 2, vietnam_services, width, label='Vietnam', color='#A23B72')

        axes[1, 0].set_title('Servizi e Trasporti (€/mese)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(services)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Indice di costo totale
        total_italy = sum([
            self.cost_of_living_data['Italy']['rent_1br_city_center'],
            self.cost_of_living_data['Italy']['utilities_basic'],
            self.cost_of_living_data['Italy']['internet'],
            self.cost_of_living_data['Italy']['transport_pass'],
            self.cost_of_living_data['Italy']['meal_inexpensive'] * 30
        ])

        total_vietnam = sum([
            self.cost_of_living_data['Vietnam']['rent_1br_city_center'],
            self.cost_of_living_data['Vietnam']['utilities_basic'],
            self.cost_of_living_data['Vietnam']['internet'],
            self.cost_of_living_data['Vietnam']['transport_pass'],
            self.cost_of_living_data['Vietnam']['meal_inexpensive'] * 30
        ])

        countries = ['Italia', 'Vietnam']
        totals = [total_italy, total_vietnam]
        colors = ['#2E86AB', '#A23B72']

        bars = axes[1, 1].bar(countries, totals, color=colors, alpha=0.8)
        axes[1, 1].set_title('Costo Vita Totale (€/mese)')
        axes[1, 1].set_ylabel('Euro')
        axes[1, 1].grid(True, alpha=0.3)

        # Aggiungere valori sulle barre
        for bar, value in zip(bars, totals):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                            f'€{value:.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return total_italy, total_vietnam

    def create_salary_comparison_chart(self):
        """Crea grafici comparativi dei salari"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Confronto Salari: Italia vs Vietnam', fontsize=16, fontweight='bold')

        # 1. Salari per tipologia
        professions = ['Minimo', 'Medio', 'Mediano', 'Sviluppatore Sr.', 'Insegnante', 'Infermiere', 'Ingegnere']
        italy_salaries = [self.salary_data['Italy'][key] for key in self.salary_data['Italy'].keys()]
        vietnam_salaries = [self.salary_data['Vietnam'][key] for key in self.salary_data['Vietnam'].keys()]

        x = np.arange(len(professions))
        width = 0.35

        axes[0].bar(x - width / 2, italy_salaries, width, label='Italia', color='#2E86AB')
        axes[0].bar(x + width / 2, vietnam_salaries, width, label='Vietnam', color='#A23B72')

        axes[0].set_title('Salari per Professione (€/mese)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(professions, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Rapporto salario/costo della vita
        total_italy, total_vietnam = 1800, 750  # Costi stimati

        salary_ratios_italy = [salary / total_italy for salary in italy_salaries]
        salary_ratios_vietnam = [salary / total_vietnam for salary in vietnam_salaries]

        axes[1].bar(x - width / 2, salary_ratios_italy, width, label='Italia', color='#2E86AB')
        axes[1].bar(x + width / 2, salary_ratios_vietnam, width, label='Vietnam', color='#A23B72')

        axes[1].set_title('Rapporto Salario/Costo Vita')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(professions, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Soglia di sopravvivenza')

        plt.tight_layout()
        plt.show()

        return italy_salaries, vietnam_salaries

    def create_inflation_analysis(self):
        """Analisi dell'inflazione"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisi Inflazione: Italia vs Vietnam', fontsize=16, fontweight='bold')

        # 1. Trend inflazione
        axes[0, 0].plot(self.years, self.inflation_data['Italy'], marker='o',
                        linewidth=2, label='Italia', color='#2E86AB')
        axes[0, 0].plot(self.years, self.inflation_data['Vietnam'], marker='s',
                        linewidth=2, label='Vietnam', color='#A23B72')
        axes[0, 0].set_title('Trend Inflazione (%)')
        axes[0, 0].set_xlabel('Anno')
        axes[0, 0].set_ylabel('Inflazione (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Inflazione media
        avg_inflation = {
            'Italia': np.mean(self.inflation_data['Italy']),
            'Vietnam': np.mean(self.inflation_data['Vietnam'])
        }

        countries = list(avg_inflation.keys())
        values = list(avg_inflation.values())
        colors = ['#2E86AB', '#A23B72']

        bars = axes[0, 1].bar(countries, values, color=colors, alpha=0.8)
        axes[0, 1].set_title('Inflazione Media 2019-2024 (%)')
        axes[0, 1].set_ylabel('Inflazione (%)')
        axes[0, 1].grid(True, alpha=0.3)

        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Impatto dell'inflazione sui salari reali
        base_salary_italy = 2500
        base_salary_vietnam = 450

        # Calcolo potere d'acquisto
        purchasing_power_italy = []
        purchasing_power_vietnam = []

        for i, year in enumerate(self.years):
            # Calcolo cumulativo dell'inflazione
            cumulative_inflation_italy = np.prod([1 + inf / 100 for inf in self.inflation_data['Italy'][:i + 1]])
            cumulative_inflation_vietnam = np.prod([1 + inf / 100 for inf in self.inflation_data['Vietnam'][:i + 1]])

            purchasing_power_italy.append(base_salary_italy / cumulative_inflation_italy)
            purchasing_power_vietnam.append(base_salary_vietnam / cumulative_inflation_vietnam)

        axes[1, 0].plot(self.years, purchasing_power_italy, marker='o',
                        linewidth=2, label='Italia', color='#2E86AB')
        axes[1, 0].plot(self.years, purchasing_power_vietnam, marker='s',
                        linewidth=2, label='Vietnam', color='#A23B72')
        axes[1, 0].set_title('Potere d\'Acquisto nel Tempo (€)')
        axes[1, 0].set_xlabel('Anno')
        axes[1, 0].set_ylabel('Potere d\'Acquisto (€)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Volatilità inflazione
        volatility_italy = np.std(self.inflation_data['Italy'])
        volatility_vietnam = np.std(self.inflation_data['Vietnam'])

        countries = ['Italia', 'Vietnam']
        volatilities = [volatility_italy, volatility_vietnam]

        bars = axes[1, 1].bar(countries, volatilities, color=colors, alpha=0.8)
        axes[1, 1].set_title('Volatilità Inflazione (Deviazione Standard)')
        axes[1, 1].set_ylabel('Volatilità (%)')
        axes[1, 1].grid(True, alpha=0.3)

        for bar, value in zip(bars, volatilities):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return avg_inflation, volatilities

    def create_purchasing_power_analysis(self):
        """Analisi del potere d'acquisto"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analisi Potere d\'Acquisto: Italia vs Vietnam', fontsize=16, fontweight='bold')

        # 1. Big Mac Index (esempio di PPP)
        big_mac_prices = {'Italia': 4.5, 'Vietnam': 2.8}
        usd_exchange_rate = {'Italia': 0.85, 'Vietnam': 0.000043}  # EUR e VND vs USD

        big_mac_ppp = {}
        for country, price in big_mac_prices.items():
            big_mac_ppp[country] = price / big_mac_prices['Italia']  # Normalizzato all'Italia

        countries = list(big_mac_ppp.keys())
        ppp_values = list(big_mac_ppp.values())
        colors = ['#2E86AB', '#A23B72']

        bars = axes[0, 0].bar(countries, ppp_values, color=colors, alpha=0.8)
        axes[0, 0].set_title('Big Mac Index (PPP)')
        axes[0, 0].set_ylabel('Rapporto PPP')
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Parità')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Ore di lavoro necessarie per beni di base
        avg_hourly_wage_italy = self.salary_data['Italy']['average_wage'] / 160  # 160 ore/mese
        avg_hourly_wage_vietnam = self.salary_data['Vietnam']['average_wage'] / 160

        basic_goods = ['Pasto ristorante', 'Cappuccino', 'Latte 1L', 'Pane 500g']
        hours_needed_italy = [
            self.cost_of_living_data['Italy']['meal_restaurant'] / avg_hourly_wage_italy,
            self.cost_of_living_data['Italy']['cappuccino'] / avg_hourly_wage_italy,
            self.cost_of_living_data['Italy']['milk_1l'] / avg_hourly_wage_italy,
            self.cost_of_living_data['Italy']['bread_500g'] / avg_hourly_wage_italy
        ]
        hours_needed_vietnam = [
            self.cost_of_living_data['Vietnam']['meal_restaurant'] / avg_hourly_wage_vietnam,
            self.cost_of_living_data['Vietnam']['cappuccino'] / avg_hourly_wage_vietnam,
            self.cost_of_living_data['Vietnam']['milk_1l'] / avg_hourly_wage_vietnam,
            self.cost_of_living_data['Vietnam']['bread_500g'] / avg_hourly_wage_vietnam
        ]

        x = np.arange(len(basic_goods))
        width = 0.35

        axes[0, 1].bar(x - width / 2, hours_needed_italy, width, label='Italia', color='#2E86AB')
        axes[0, 1].bar(x + width / 2, hours_needed_vietnam, width, label='Vietnam', color='#A23B72')

        axes[0, 1].set_title('Ore di Lavoro per Beni di Base')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(basic_goods, rotation=45)
        axes[0, 1].set_ylabel('Ore di lavoro')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Costo affitto vs salario
        rent_to_income_italy = (self.cost_of_living_data['Italy']['rent_1br_city_center'] /
                                self.salary_data['Italy']['average_wage']) * 100
        rent_to_income_vietnam = (self.cost_of_living_data['Vietnam']['rent_1br_city_center'] /
                                  self.salary_data['Vietnam']['average_wage']) * 100

        rent_ratios = [rent_to_income_italy, rent_to_income_vietnam]

        bars = axes[1, 0].bar(countries, rent_ratios, color=colors, alpha=0.8)
        axes[1, 0].set_title('Affitto come % del Salario')
        axes[1, 0].set_ylabel('Percentuale (%)')
        axes[1, 0].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Soglia consigliata 30%')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        for bar, value in zip(bars, rent_ratios):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Indice di accessibilità generale
        # Calcolo indice composito (più basso = più accessibile)
        affordability_italy = np.mean([
            rent_to_income_italy / 30,  # Normalizzato alla soglia del 30%
            np.mean(hours_needed_italy),
            self.cost_of_living_data['Italy']['meal_inexpensive'] / (avg_hourly_wage_italy)
        ])

        affordability_vietnam = np.mean([
            rent_to_income_vietnam / 30,
            np.mean(hours_needed_vietnam),
            self.cost_of_living_data['Vietnam']['meal_inexpensive'] / (avg_hourly_wage_vietnam)
        ])

        affordability_scores = [affordability_italy, affordability_vietnam]

        bars = axes[1, 1].bar(countries, affordability_scores, color=colors, alpha=0.8)
        axes[1, 1].set_title('Indice di Accessibilità\n(più basso = più accessibile)')
        axes[1, 1].set_ylabel('Punteggio')
        axes[1, 1].grid(True, alpha=0.3)

        for bar, value in zip(bars, affordability_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return rent_ratios, affordability_scores

    def create_summary_report(self):
        """Genera report riassuntivo"""
        print("=" * 60)
        print("REPORT RIASSUNTIVO: ITALIA vs VIETNAM")
        print("=" * 60)

        # Costi totali
        total_italy = sum([
            self.cost_of_living_data['Italy']['rent_1br_city_center'],
            self.cost_of_living_data['Italy']['utilities_basic'],
            self.cost_of_living_data['Italy']['internet'],
            self.cost_of_living_data['Italy']['transport_pass'],
            self.cost_of_living_data['Italy']['meal_inexpensive'] * 30
        ])

        total_vietnam = sum([
            self.cost_of_living_data['Vietnam']['rent_1br_city_center'],
            self.cost_of_living_data['Vietnam']['utilities_basic'],
            self.cost_of_living_data['Vietnam']['internet'],
            self.cost_of_living_data['Vietnam']['transport_pass'],
            self.cost_of_living_data['Vietnam']['meal_inexpensive'] * 30
        ])

        print(f"\n1. COSTO DELLA VITA MENSILE:")
        print(f"   Italia: €{total_italy:.0f}")
        print(f"   Vietnam: €{total_vietnam:.0f}")
        print(f"   Differenza: {((total_italy - total_vietnam) / total_vietnam * 100):.1f}% più caro in Italia")

        print(f"\n2. SALARI MEDI:")
        print(f"   Italia: €{self.salary_data['Italy']['average_wage']:.0f}")
        print(f"   Vietnam: €{self.salary_data['Vietnam']['average_wage']:.0f}")
        print(
            f"   Rapporto: {(self.salary_data['Italy']['average_wage'] / self.salary_data['Vietnam']['average_wage']):.1f}x")

        print(f"\n3. INFLAZIONE MEDIA (2019-2024):")
        print(f"   Italia: {np.mean(self.inflation_data['Italy']):.1f}%")
        print(f"   Vietnam: {np.mean(self.inflation_data['Vietnam']):.1f}%")

        print(f"\n4. POTERE D'ACQUISTO:")
        italy_surplus = self.salary_data['Italy']['average_wage'] - total_italy
        vietnam_surplus = self.salary_data['Vietnam']['average_wage'] - total_vietnam
        print(f"   Surplus mensile Italia: €{italy_surplus:.0f}")
        print(f"   Surplus mensile Vietnam: €{vietnam_surplus:.0f}")

        print(f"\n5. RACCOMANDAZIONI:")
        if vietnam_surplus > italy_surplus:
            print("   - Il Vietnam offre un migliore rapporto costo/beneficio")
        else:
            print("   - L'Italia offre salari più alti che compensano i costi maggiori")

        print("   - Considerare differenze di qualità dei servizi")
        print("   - Valutare opportunità di carriera a lungo termine")
        print("   - Analizzare stabilità economica e politica")

        print("=" * 60)

    def run_complete_analysis(self):
        """Esegue l'analisi completa"""
        print("Iniziando analisi comparativa Italia vs Vietnam...")
        print("Generando grafici step by step...\n")

        # Step 1: Confronto costi
        print("Step 1: Analisi costi della vita")
        self.create_cost_comparison_chart()

        # Step 2: Confronto salari
        print("\nStep 2: Analisi salari")
        self.create_salary_comparison_chart()

        # Step 3: Analisi inflazione
        print("\nStep 3: Analisi inflazione")
        self.create_inflation_analysis()

        # Step 4: Potere d'acquisto
        print("\nStep 4: Analisi potere d'acquisto")
        self.create_purchasing_power_analysis()

        # Step 5: Report finale
        print("\nStep 5: Report riassuntivo")
        self.create_summary_report()

        print("\nAnalisi completata!")


# Utilizzo
if __name__ == "__main__":
    # Creazione dell'analisi
    analysis = CostOfLivingAnalysis()

    # Esecuzione dell'analisi completa
    analysis.run_complete_analysis()

    # Nota: Per utilizzare API reali, sostituire i dati di esempio con:
    # - API Numbeo per costi della vita
    # - API World Bank per dati economici
    # - API OECD per statistiche ufficiali
    # - API ExchangeRate per tassi di cambio in tempo reale

    print("\n" + "=" * 60)
    print("INSTALLAZIONE DIPENDENZE NECESSARIE:")
    print("pip install requests pandas matplotlib seaborn numpy")
    print("=" * 60)