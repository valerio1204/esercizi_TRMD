"""
Fit dei nuovi contagi usando emcee campionando direttamente N (N_tot).
- Modello: Logistica e Gompertz cumulativa, t0 fissato (giorno 0 = 24 febbraio)
- Dati: dal 24 febbraio al 30 marzo per il fit (35 giorni)
- Previsione: confronto con i primi 100 giorni dall'inizio (24 febbraio)
- Priors:
    N ~ Uniform(Nmin, Nmax)
    k ~ Uniform(kmin, kmax)
- Output:
    * Stima dei parametri (mediana + intervallo credibile 16–84%)
    * Confronto predizioni (100 giorni) vs osservati (100 giorni)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import emcee
from scipy.special import gammaln

# -----------------------------
# Modelli cumulativi e utilità
# -----------------------------

# qui userò t come array di giorni (0,1,2,...) e i nuovi contagi giornalieri li ricavo come differenza tra due array con t traslato di 1
# np mi permette di fare operazioni con array in modo vettoriale, senza dover usare loop espliciti

def logistic_cum(t, N, k, t0=0.0):                
    return N * (1.0 / (1.0 +  N * np.exp( -k * (t - t0))))


def gompertz_cum(t, N, k, t0=0.0):
    return N * np.exp(-np.log(N) * np.exp(-k * (t - t0)))


def daily_from_cum_func(cum_func, t, N, k, t0=0.0):  # qui tra gli argomenti come cum_func posso avere logistica o gompertz
    t = np.asarray(t)                                # creo un array numpy a partire da t
    cum_t = cum_func(t, N, k, t0)
    cum_t1 = cum_func(t + 1.0, N, k, t0)
    daily = cum_t1 - cum_t                           # calcolo i nuovi casi giornalieri come differenza
    return np.clip(daily, 1e-9, None)                # evito valori <= 0


def poisson_loglike(mu, obs):
    mu = np.clip(mu, 1e-12, None)  # evito valori <= 0 per i val. di aspettazione senno log(0)
    obs = np.asarray(obs)          # assicuro che obs (ovvero i parametri liberi) sia un array numpy
    return np.sum(obs * np.log(mu) - mu - gammaln(obs + 1.0))

# -----------------------------
# Priors: N uniforme e k uniforme
# -----------------------------

# definisco i priors di N e k come distribuz uniformi con N compreso tra Nmin(contagi effettivamente registrati fino al 30/03) e Nmax(calcolato con kmax) e k tra kmin e kmax

def build_priors_from_data(obs_daily_window, factor_max_N=1000.0):
    cum = np.cumsum(obs_daily_window)
    Nmin = cum[-1]                                # cum è un array numpy, cum[-1] è l'ultimo elemento, ovvero il numero totale di contagi registrati
    Nmax = Nmin * factor_max_N
    return dict(Nmin=float(Nmin), Nmax=float(Nmax), kmin=1e-3, kmax=2.0, t0=0.0)


def log_prior(theta, priors):               # priors è un dizionario che contiene i valori min e max di N e k
    N, k = theta
    if not (priors['Nmin'] < N < priors['Nmax']):
        return -np.inf
    if not (priors['kmin'] < k < priors['kmax']):
        return -np.inf
    return 0.0                              # log(1) = 0, quindi se i parametri sono nei limiti il prior è costante, altrimenti è 0


def log_probability(theta, t, obs, cum_model_func, priors):           # lavoro con i log percheè sto usando il loglikelihood
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    N, k = theta
    mu = daily_from_cum_func(cum_model_func, t, N, k, priors.get('t0', 0.0))
    ll = poisson_loglike(mu, obs)
    return lp + ll                                               # somma di log prior e log likelihood, quindi ho log posterior

# -----------------------------
# Fit + emcee wrapper
# -----------------------------

def fit_with_emcee_N(t, obs, cum_model_func, priors, nwalkers=128, nsteps=10000, discard_burnin=2000):
    cum = np.cumsum(obs)
    N_init = cum[-1] * 2.0
    k_init = 0.2
    initial = np.array([N_init, k_init])

    nll = lambda x: -log_probability(x, t, obs, cum_model_func, priors)        #minimizzo la neg log posterior per trovare un buon punto di partenza per emcee
    bounds = [(priors['Nmin']*1.0000001, priors['Nmax']*0.9999999), (priors['kmin'], priors['kmax'])]   # faccio eseguire minimize all'interno dei bordi
    try:
        sol = minimize(nll, initial, bounds=bounds)  # il risultato è un oggetto con varie informazioni, tra cui x che contiene le coordinate del minimo
        x0 = sol.x
    except Exception:
        x0 = initial

    ndim = len(x0)                                 # numero di parametri liberi (N e k, quindi 2)
    rng = np.random.default_rng(42)                # uso un generatore di numeri casuali con seed fisso per riproducibilità
    pos = x0 + 1e-4 * x0 * rng.standard_normal((nwalkers, ndim)) + 1e-6 * rng.standard_normal((nwalkers, ndim))
    # qui sopra ho aggiunto un piccolo rumore gaussiano intorno al punto di minimo per inizializzare i walker (il primo termine è proporzionale a x0 per avere un rumore consono
    # rispetto alla scala dei parametri, il secondo termine è un rumore molto piccolo per evitare di avere tutti i walker nello stesso punto se x0 ha un valore vicino a 0)


    for i in range(nwalkers):                                      # assicuro che tutti i walker partano all'interno dei limiti dei priors
        if not (priors['Nmin'] < pos[i,0] < priors['Nmax']):
            pos[i,0] = rng.uniform(priors['Nmin'], priors['Nmax'])
        if not (priors['kmin'] < pos[i,1] < priors['kmax']):
            pos[i,1] = rng.uniform(priors['kmin'], priors['kmax'])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t, obs, cum_model_func, priors))    # args sono gli argomenti addizionali da passare a log_probability
    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=discard_burnin, flat=True)    # flat=True appiattisce la catena in un array 2D (nwalkers*nsteps, ndim)
    return sampler, samples, x0                                       # ritorna la catena completa dei campioni (dopo aver scartato il burn-in) e il punto di partenza x0

# -----------------------------
# Posterior predictive
# -----------------------------

# qui ricavo l'incertezza lavorando sui modelli che si ricavano dai vari samples dei parametri, qui uso n_samples sul totale dei samples per non appesantire troppo i calcoli
# poi userò la mediana (50 percentile) e l'intervallo 16-84 percentili per rappresentare l'incertezza
def posterior_predictive(samples, t_pred, cum_model_func, n_samples=5000):         
    rng = np.random.default_rng(123)
    idx = rng.choice(np.arange(len(samples)), size=n_samples, replace=False)      # scelgo n_samples indici casuali tra quelli disponibili nei samples
    preds = []
    for ii in idx:
        N, k = samples[ii]
        mu = daily_from_cum_func(cum_model_func, t_pred, N, k)
        preds.append(mu)                         # creo una lista di array numpy, ogni array è la previsione dei nuovi contagi per un set di parametri (N,k) per i vari giorni
    preds = np.array(preds)                      # da lista ad array 2d numpy 
    low = np.percentile(preds, 16, axis=0)
    med = np.percentile(preds, 50, axis=0)
    high = np.percentile(preds, 84, axis=0)
    return med, low, high, preds

# -----------------------------
# Utility: stampa riassunto parametri
# -----------------------------
def summarize_parameters(samples, names=['N','k']):
    med = np.median(samples, axis=0)                            # <-- l'asse 0 corrisponde ai samples dei parametri N e k raggruppati come array di dim 2
    low = np.percentile(samples, 16, axis=0)                    # lungo l'asse 0 che è lunga nwalkers*nsteps
    high = np.percentile(samples, 84, axis=0)
    for n,m,l,h in zip(names, med, low, high):                  # zip accoppia gli i esimi elementi di più liste, in questo caso names, med, low, high
        print(f"{n}: {m:.3g}  (16%={l:.3g}, 84%={h:.3g})")
    return med, low, high

# -----------------------------
# Esecuzione
# -----------------------------

region_name = 'Friuli Venezia Giulia'  
csv_path = 'https://github.com/valerio1204/esercizi_TRMD/blob/main/dpc-covid19-ita-regioni.csv?raw=true'  # path del file csv con i dati
start_date = '2020-02-24'   # inizio analisi (giorno 0)
fit_end = '2020-03-30'      # fine finestra per il fit
compare_days = 100          # confronto con 100 giorni
# Carica dati e prepara index in modo robusto
df = pd.read_csv(csv_path, parse_dates=['data'])
df_reg = df[df['denominazione_regione'] == region_name].sort_values('data').set_index('data')
 # Assicuriamoci che l'indice(ricordo che indica l'indice delle righe) sia datetime (per lavorare meglio con array)
df_reg.index = pd.to_datetime(df_reg.index)
    
start_date = pd.to_datetime(start_date)        # converto da stringa a datetime
end_compare = start_date + pd.Timedelta(days=compare_days-1)        # intervallo di 100 giorni arriva al 99esimo giorno
df_100 = df_reg.loc[start_date : end_compare]

# Se non ci sono abbastanza giorni, riduci compare_days ai giorni effettivamente disponibili
if len(df_100) < compare_days:
        print(f"Attenzione: richiesti {compare_days} giorni a partire da {start_date.date()}, ma ne sono disponibili solo {len(df_100)}. Ridurrò compare_days a {len(df_100)}.")
        compare_days = len(df_100)
        # ricomputa end_compare coerentemente
        end_compare = start_date + pd.Timedelta(days=compare_days-1)
        df_100 = df_reg.loc[start_date : end_compare]                 # prende i dati nei giorni di interesse

obs_daily_100 = df_100['nuovi_positivi'].values.astype(int)       # seleziona i nuovi contagi nei giorni di interesse

# Finestra per il fit: da start_date fino a fit_end, cioè la data di fine fit trasformata in datetime
df_fit = df_reg.loc[start_date : pd.to_datetime(fit_end)]
if len(df_fit) == 0:
        raise ValueError(f"Non ci sono dati tra {start_date.date()} e {fit_end} per {region_name}.")
obs_daily_fit = df_fit['nuovi_positivi'].values.astype(int)      # seleziona i nuovi contagi nella finestra di fit
t = np.arange(len(obs_daily_fit))

priors = build_priors_from_data(obs_daily_fit, factor_max_N=5.0)
print('Priors usati:', priors)

# Fit logistica
print('\n--- Fit LOGISTICA ---')
sampler_log, samples_log, x0_log = fit_with_emcee_N(t, obs_daily_fit, logistic_cum, priors,
                                                      nwalkers=32, nsteps=2000, discard_burnin=500)
summarize_parameters(samples_log)

# Fit Gompertz
print('\n--- Fit GOMPERTZ ---')
sampler_gom, samples_gom, x0_gom = fit_with_emcee_N(t, obs_daily_fit, gompertz_cum, priors,
                                                      nwalkers=32, nsteps=2000, discard_burnin=500)
summarize_parameters(samples_gom)

# Predizione e confronto con 100 giorni  (considero le curve mediane e l'intervallo 16-84%, usando un sample di 1000 su tutti i samples)
t_pred = np.arange(compare_days)
med_log, low_log, high_log, _ = posterior_predictive(samples_log, t_pred, logistic_cum, n_samples=1000)
med_gom, low_gom, high_gom, _ = posterior_predictive(samples_gom, t_pred, gompertz_cum, n_samples=1000)

# -----------------------------
# Plots
# -----------------------------

# per le date nel grafico che va dal 2020 al 2025 (uso sempre datetime)
start = pd.Timestamp("2020-02-24")
end = pd.Timestamp("2025-01-08")
dates = pd.date_range(start=start, end=end, freq="D")
dates = pd.to_datetime(dates)

obs_daily_total = df_reg['nuovi_positivi'].values.astype(int)  # nuovi contagi totali registrati

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, obs_daily_total, marker='.', linestyle='-', label='Nuovi positivi giornalieri')
ax.set_title(f"Nuovi positivi per giorno - {region_name}")
ax.set_xlabel("Data")
ax.set_ylabel("Nuovi positivi")
ax.legend()
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()     # impedisce che il grafico esca dai bordi della figura
plt.show()



plt.figure(figsize=(12,7))
plt.plot(np.arange(compare_days-1), obs_daily_100, 'k.', label='osservati (100 giorni)')
plt.plot(t, obs_daily_fit, 'ro', label='usati per il fit (24 feb–30 mar)')

plt.plot(t_pred, med_log, '-', label='logistica mediana')
plt.fill_between(t_pred, low_log, high_log, alpha=0.3)

plt.plot(t_pred, med_gom, '--', label='gompertz mediana')
plt.fill_between(t_pred, low_gom, high_gom, alpha=0.2)

plt.xlabel(f'giorni dall\'inizio (0 = {start_date})')
plt.ylabel('nuovi casi giornalieri')
plt.title(f"Confronto predizioni vs osservati ({region_name})")
plt.legend()
plt.show()


def plot_walkers(sampler, model_name='model', burn=0):
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    param_names = ['N', 'k']

    # Trace plot
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5*ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i] if ndim > 1 else axes
        ax.plot(chain[:, :, i], color='k', alpha=0.35, linewidth=0.6)
        ax.set_ylabel(param_names[i])
    axes[-1].set_xlabel('step')
    plt.suptitle(f"Trace plot walker positions – {model_name}")
    plt.show()

    # Snapshots scatter (N vs k) a vari step
    steps = np.linspace(burn, nsteps-1, 6, dtype=int)  # 6 snapshot equispaziati
    colors = plt.cm.viridis(np.linspace(0,1,len(steps)))
    plt.figure(figsize=(8,6))
    for s,c in zip(steps, colors):
        pts = chain[s, :, :]
        plt.scatter(pts[:,0], pts[:,1], alpha=0.7, s=30, color=c, label=f'step {s}')
    plt.xlabel('N'); plt.ylabel('k')
    plt.title(f"Walker positions (snapshots) – {model_name}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

# Disegna per logistica e Gompertz
plot_walkers(sampler_log, model_name='logistica')
plot_walkers(sampler_gom,  model_name='gompertz')

#-----------------------------
#grafici cumulativi
#-----------------------------

plt.figure(figsize=(12,7))

# Cumulata osservata nei 100 giorni
obs_cum_100 = np.cumsum(obs_daily_100)
plt.plot(np.arange(compare_days-1), obs_cum_100, 'k.', label='osservati cumulativi (100 giorni)')

# Cumulata osservata usata per il fit (24 feb – 30 mar)
obs_cum_fit = np.cumsum(obs_daily_fit)
plt.plot(t, obs_cum_fit, 'ro', label='usati per il fit (cumulativi)')

# Cumulata predetta logistica
cum_log_med = np.cumsum(med_log)
plt.plot(t_pred, cum_log_med, '-', label='logistica mediana')
plt.fill_between(t_pred,
                 np.cumsum(low_log),
                 np.cumsum(high_log),
                 alpha=0.3)

# Cumulata predetta Gompertz
cum_gom_med = np.cumsum(med_gom)
plt.plot(t_pred, cum_gom_med, '--', label='gompertz mediana')
plt.fill_between(t_pred,
                 np.cumsum(low_gom),
                 np.cumsum(high_gom),
                 alpha=0.2)

plt.xlabel(f'giorni dall\'inizio (0 = {start_date})')
plt.ylabel('casi cumulativi')
plt.title(f"Confronto predizioni cumulative vs osservati ({region_name})")
plt.legend()
plt.show()



