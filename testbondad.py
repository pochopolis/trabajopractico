import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv('bmi.csv')

df_adolescentes_argentina = df[(df['edad'] == 'adolescente') & (df['país'] == 'argentina')]

datos_altura_argentina = df_adolescentes_argentina['altura (cm)']

media_altura_argentina = np.mean(datos_altura_argentina)
desvio_altura_argentina = np.std(datos_altura_argentina)

num_bins = 10
bins = np.linspace(np.min(datos_altura_argentina), np.max(datos_altura_argentina), num_bins + 1)
observed_freq, _ = np.histogram(datos_altura_argentina, bins)

cdf_values = stats.norm.cdf(bins, loc=media_altura_argentina, scale=desvio_altura_argentina)
expected_freq = np.diff(cdf_values) * len(datos_altura_argentina)

expected_freq *= observed_freq.sum() / expected_freq.sum()

chi2_stat, p_valor = stats.chisquare(observed_freq, expected_freq, ddof=2)

nivel_significancia = 0.05

if p_valor < nivel_significancia:
    print(f"Se rechaza la H0 (p-valor = {p_valor:.4f})")
else:
    print(f"No se rechaza la H0 (p-valor = {p_valor:.4f})")

