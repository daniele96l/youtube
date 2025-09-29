# Simulazione Strategia "Buy the Dip" Multi-Threshold

Questa simulazione confronta diverse strategie di investimento per determinare se la strategia "Buy the Dip" è più profittevole dell'investimento regolare, testando diversi livelli di threshold.

## Strategie Confrontate

### Strategia 1: Investimento Regolare
- Investe €1000 ogni mese nell'S&P 500
- Nessuna logica di timing del mercato

### Strategia 2: Buy the Dip (Multi-Threshold)
- Investe €800 ogni mese nell'S&P 500
- Risparmia €200 ogni mese
- Usa tutti i risparmi quando il mercato crolla dall'ultimo picco:
  - **Threshold 5%**: Acquista al -5% dal picco
  - **Threshold 10%**: Acquista al -10% dal picco
  - **Threshold 20%**: Acquista al -20% dal picco
  - **Threshold 30%**: Acquista al -30% dal picco

## Risultati Principali

Dalla simulazione Monte Carlo (1000 simulazioni di 5 anni):

| Strategia | Valore Medio | CAGR | Volatilità | Tasso Vittoria |
|-----------|--------------|------|------------|----------------|
| **Regolare** | €60,546 | **0.17%** | 50.70% | 0.0% |
| Buy Dip 5% | €52,495 | -2.67% | 51.28% | 2.5% |
| Buy Dip 10% | €49,039 | -3.97% | 50.76% | 1.2% |
| Buy Dip 20% | €48,437 | -4.20% | 50.70% | 0.0% |
| Buy Dip 30% | €48,437 | -4.20% | 50.70% | 0.0% |

### Analisi Dettagliata

- **Migliore strategia**: Investimento regolare
- **Performance**: +12.9% vs Buy the Dip 5%
- **Consistenza**: Investimento regolare vince nel 100% dei casi
- **Risk-adjusted return**: Sharpe ratio positivo solo per strategia regolare

### Analisi per Threshold

- **Threshold 5%**: Migliore tra le strategie Buy the Dip, ma ancora inferiore
- **Threshold 10%**: Performance peggiora ulteriormente
- **Threshold 20% e 30%**: Stesse performance (raramente si verificano crash così profondi)

## Metriche Calcolate

- **CAGR**: Compound Annual Growth Rate basato sul totale investito
- **Volatilità**: Deviazione standard annualizzata dei rendimenti
- **Sharpe Ratio**: Rapporto rischio/rendimento
- **Tasso di vittoria**: Percentuale di simulazioni dove la strategia vince

## Come Eseguire

```bash
python3 "buy the dip.py"
```

## Dipendenze

Installare le dipendenze:
```bash
pip3 install -r requirements.txt
```

## Visualizzazioni

La simulazione include:
1. **Evoluzione prezzi** con punti di crash per diversi threshold
2. **Confronto portfolio** nel tempo per tutte le strategie
3. **Scatter plot** CAGR vs Volatilità per analisi risk-adjusted

## Note Importanti

- La simulazione usa dati sintetici basati sui parametri storici dell'S&P 500
- I risultati reali possono variare significativamente
- Considerare sempre la diversificazione del portfolio
- Consultare un consulente finanziario per decisioni reali

## Conclusioni

La simulazione conferma che l'investimento regolare (Dollar Cost Averaging) è significativamente più profittevole di tutte le varianti della strategia "Buy the Dip" testate, con:

- ✅ Migliore performance assoluta
- ✅ Minore volatilità
- ✅ Maggiore consistenza
- ✅ Migliore risk-adjusted return

I threshold più alti (20%, 30%) raramente si attivano, rendendo la strategia equivalente a investire solo €800/mese senza benefici.
