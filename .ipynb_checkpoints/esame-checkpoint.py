import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import difflib

path = "/mnt/data/dpc-covid19-ita-regioni.csv"
df = pd.read_csv(path, parse_dates=["data"])

regioni = sorted(df["denominazione_regione"].unique())
scelta = "Lombardia"   # <-- cambia qui il nome della regione che vuoi analizzare
if scelta not in regioni:
    suggerimenti = difflib.get_close_matches(scelta, regioni, n=6, cutoff=0.4)
    print(f"Regione '{scelta}' non trovata. Suggerimento: {suggerimenti}")
    scelta = regioni[0]  # fallback automatico

df_reg = df[df["denominazione_regione"] == scelta].copy()
df_reg["data_giorno"] = df_reg["data"].dt.date

serie_giornaliera = df_reg.groupby("data_giorno")["nuovi_positivi"].sum().sort_index()
media_mobile_7 = serie_giornaliera.rolling(window=7, center=True).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(serie_giornaliera.index, serie_giornaliera.values, marker='.', linestyle='-', label='Nuovi positivi giornalieri')
ax.plot(media_mobile_7.index, media_mobile_7.values, linestyle='--', label='Media mobile 7 giorni')
ax.set_title(f"Nuovi positivi per giorno - {scelta}")
ax.set_xlabel("Data")
ax.set_ylabel("Nuovi positivi")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()
