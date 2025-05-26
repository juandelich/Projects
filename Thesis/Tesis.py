import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

# -------------------------------
# Data Loading
# -------------------------------
data = pd.read_excel("Bonos VF.xlsx", sheet_name="Paises", header=0, index_col=0, parse_dates=True)
data = data.apply(pd.to_numeric, errors='coerce')
print(data.columns)

Factores_externos = pd.read_excel("Bonos VF.xlsx", sheet_name="Indice HY", header=0, index_col=0, parse_dates=True)
Factores_externos = Factores_externos.apply(pd.to_numeric, errors='coerce')
Factores_externos.index = pd.to_datetime(Factores_externos.index, errors="coerce")

# -------------------------------
# Data Preparation
# -------------------------------
bond_daily_log_returns = np.log(data / data.shift(1))
bond_weekly_log_returns = np.log(data.resample("W").last() / data.resample("W").last().shift(1))
bond_monthly_log_returns = np.log(data.resample("M").last() / data.resample("M").last().shift(1))

bond_daily_log_returns.dropna(inplace=True)
bond_weekly_log_returns.dropna(inplace=True)
bond_monthly_log_returns.dropna(inplace=True)

bond_daily_log_returns = bond_daily_log_returns.loc["2014-01-01":"2024-12-31"]
bond_weekly_log_returns = bond_weekly_log_returns.loc["2014-01-01":"2024-12-31"]
bond_monthly_log_returns = bond_monthly_log_returns.loc["2014-01-01":"2024-12-31"]

FE_daily_log_returns = np.log(Factores_externos / Factores_externos.shift(1))
FE_weekly_log_returns = np.log(Factores_externos.resample("W").last() / Factores_externos.resample("W").last().shift(1))
FE_monthly_log_returns = np.log(Factores_externos.resample("M").last() / Factores_externos.resample("M").last().shift(1))

FE_daily_log_returns.dropna(inplace=True)
FE_weekly_log_returns.dropna(inplace=True)
FE_monthly_log_returns.dropna(inplace=True)

FE_daily_log_returns = FE_daily_log_returns.loc["2014-01-01":"2024-12-31"]
FE_weekly_log_return = FE_weekly_log_returns.loc["2014-01-01":"2024-12-31"]
FE_monthly_log_returns = FE_monthly_log_returns.loc["2014-01-01":"2024-12-31"]

print(FE_daily_log_returns.head())

# -------------------------------
# Non-Adjusted Correlation Calculation
# -------------------------------
non_adjusted_daily_log_returns_corr = bond_daily_log_returns.corr()
non_adjusted_weekly_log_returns_corr = bond_weekly_log_returns.corr()
non_adjusted_monthly_log_returns_corr = bond_monthly_log_returns.corr()

daily_non_adjusted_rolling_corr = bond_daily_log_returns.rolling(window=60).corr()
daily_non_adjusted_average_corr_df = pd.DataFrame(index=bond_daily_log_returns.index, columns=["Average_Correlation"])
for date in bond_daily_log_returns.index:
    corr_matrix = daily_non_adjusted_rolling_corr.loc[date]
    if not corr_matrix.isnull().values.all():
        upper_triangle = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
        daily_non_adjusted_average_corr_df.loc[date, "Average_Correlation"] = upper_triangle.mean().mean()
daily_non_adjusted_average_corr_df.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(daily_non_adjusted_average_corr_df.index, daily_non_adjusted_average_corr_df["Average_Correlation"], label="Average Correlation")
plt.title("Average Correlation Across Countries (Rolling 60-Day Window)")
plt.xlabel("Fecha")
plt.ylabel("Correlación Promedio")
plt.grid(True)
plt.legend()
plt.show()

weekly_non_adjusted_rolling_corr = bond_weekly_log_returns.rolling(window=12).corr()
weekly_non_adjusted_average_corr_df = pd.DataFrame(index=bond_weekly_log_returns.index, columns=["Average_Correlation"])
for date in bond_weekly_log_returns.index:
    corr_matrix = weekly_non_adjusted_rolling_corr.loc[date]
    if not corr_matrix.isnull().values.all():
        upper_triangle = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
        weekly_non_adjusted_average_corr_df.loc[date, "Average_Correlation"] = upper_triangle.mean().mean()
weekly_non_adjusted_average_corr_df.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(weekly_non_adjusted_average_corr_df.index, weekly_non_adjusted_average_corr_df["Average_Correlation"], label="Average Correlation")
plt.title("Figura 1. Promedio de Correlación entre Paises (Ventana de Tres Meses)")
plt.xlabel("Fecha")
plt.ylabel("Correlación promedio")
plt.grid(True)
plt.show()

corr_matrix.to_csv("corr_matrix.csv")
print(corr_matrix)

# -------------------------------
# Daily Adjustment with External Factors
# -------------------------------
FE_daily_log_returns.index = pd.to_datetime(FE_daily_log_returns.index, errors="coerce")
bond_daily_log_returns.index = pd.to_datetime(bond_daily_log_returns.index, errors="coerce")
data = pd.merge(bond_daily_log_returns, FE_daily_log_returns, left_index=True, right_index=True, how="inner")
daily_country_returns = bond_daily_log_returns.columns
daily_external_factors = FE_daily_log_returns.columns
daily_residuals_df = pd.DataFrame(index=data.index, columns=daily_country_returns)
for country in daily_country_returns:
    y = data[country]
    X = data[daily_external_factors]
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=60)
    rolling_results = model.fit()
    rolling_params = rolling_results.params
    predicted_values = (X * rolling_params).sum(axis=1)
    daily_residuals_df[country] = y - predicted_values
daily_residuals_df.dropna(inplace=True)
print(daily_residuals_df.head())

rolling_corr = daily_residuals_df.rolling(window=60).corr()
average_corr_df = pd.DataFrame(index=daily_residuals_df.index, columns=["Average_Correlation"])
for date in daily_residuals_df.index:
    corr_matrix = rolling_corr.loc[date]
    if not corr_matrix.isnull().values.all():
        upper_triangle = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
        average_corr_df.loc[date, "Average_Correlation"] = upper_triangle.mean().mean()
average_corr_df.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(average_corr_df.index, average_corr_df["Average_Correlation"], label="Adjusted Average Correlation")
plt.plot(daily_non_adjusted_average_corr_df.index, daily_non_adjusted_average_corr_df["Average_Correlation"], label="Non-Adjusted Average Correlation")
plt.title("Average Correlation of Residuals Across Countries (Rolling 60-Day Window)")
plt.xlabel("Date")
plt.ylabel("Average Correlation")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# Weekly Adjustment with External Factors
# -------------------------------
FE_weekly_log_returns.index = pd.to_datetime(FE_weekly_log_returns.index, errors="coerce")
bond_weekly_log_returns.index = pd.to_datetime(bond_weekly_log_returns.index, errors="coerce")
data = pd.merge(bond_weekly_log_returns, FE_weekly_log_returns, left_index=True, right_index=True, how="inner")
weekly_country_returns = bond_weekly_log_returns.columns
weekly_external_factors = FE_weekly_log_returns.columns
weekly_residuals_df = pd.DataFrame(index=data.index, columns=weekly_country_returns)
for country in weekly_country_returns:
    y = data[country]
    X = data[weekly_external_factors]
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=60)
    rolling_results = model.fit()
    rolling_params = rolling_results.params
    predicted_values = (X * rolling_params).sum(axis=1)
    weekly_residuals_df[country] = y - predicted_values
weekly_residuals_df.dropna(inplace=True)
rolling_corr = weekly_residuals_df.rolling(window=12).corr()
average_corr_df = pd.DataFrame(index=weekly_residuals_df.index, columns=["Average_Correlation"])
for date in weekly_residuals_df.index:
    corr_matrix = rolling_corr.loc[date]
    if not corr_matrix.isnull().values.all():
        upper_triangle = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
        average_corr_df.loc[date, "Average_Correlation"] = upper_triangle.mean().mean()
average_corr_df.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(average_corr_df.index, average_corr_df["Average_Correlation"], label="Correlación Promedio Ajustada", color="Grey")
plt.plot(weekly_non_adjusted_average_corr_df.index, weekly_non_adjusted_average_corr_df["Average_Correlation"], label="Correlación Promedio No Ajustada", color="darkblue")
plt.title("Figura 2. Promedio de Correlación entre Paises (Ventana de Tres Meses)")
plt.xlabel("Fecha")
plt.ylabel("Correlación promedio")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# PCA and Clustering
# -------------------------------
weekly_residuals_PCA = weekly_residuals_df
weekly_residuals_PCA.index = pd.to_datetime(weekly_residuals_PCA.index)
weekly_residuals_PCA['year'] = weekly_residuals_PCA.index.year
df_2022 = weekly_residuals_PCA[weekly_residuals_PCA['year'] == 2017].drop(columns='year')
df_2022_std = (df_2022 - df_2022.mean()) / df_2022.std()
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(df_2022_std)
print("Explained variance ratio:", pca.explained_variance_ratio_)
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df_2022_std.columns, columns=['PC1', 'PC2', 'PC3'])
print(loadings_df)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(loadings_df)
loadings_df['Cluster'] = clusters
print(loadings_df)
plt.figure(figsize=(8,6))
scatter = plt.scatter(loadings_df['PC1'], loadings_df['PC2'], c=loadings_df['Cluster'], cmap='viridis', s=100)
for country, row in loadings_df.iterrows():
    plt.text(row['PC1']+0.01, row['PC2']+0.01, country, fontsize=9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Country Clusters Based on PCA Loadings (2022)')
plt.colorbar(scatter, label='Cluster')
plt.show()

results = {}
for year, group in weekly_residuals_PCA.groupby('year'):
    group = group.drop(columns='year')
    group_std = (group - group.mean()) / group.std()
    pca = PCA(n_components=2)
    pca.fit(group_std)
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, index=group_std.columns, columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(loadings_df)
    loadings_df['Cluster'] = clusters
    results[year] = {
        'loadings': loadings_df[['PC1', 'PC2']],
        'explained_variance': pca.explained_variance_ratio_
    }
summary_data = []
for year, info in results.items():
    loadings_df = info['loadings']
    explained = info['explained_variance']
    for idx, comp in enumerate(loadings_df.columns):
        avg_loading = loadings_df[comp].mean()
        summary_data.append({
            'Year': year,
            'Component': comp,
            'Average_Loading': avg_loading,
            'Explained_Variance_Ratio': explained[idx]
        })
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by='Year').reset_index(drop=True)
print(summary_df)
print(summary_df)

weekly_residuals_PCA = bond_weekly_log_returns
weekly_residuals_PCA.index = pd.to_datetime(weekly_residuals_PCA.index)
weekly_residuals_PCA['year'] = weekly_residuals_PCA.index.year
df_2022 = weekly_residuals_PCA[weekly_residuals_PCA['year'] == 2017].drop(columns='year')
df_2022_std = (df_2022 - df_2022.mean()) / df_2022.std()
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(df_2022_std)
print("Explained variance ratio:", pca.explained_variance_ratio_)
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df_2022_std.columns, columns=['PC1', 'PC2', 'PC3'])
print(loadings_df)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(loadings_df)
loadings_df['Cluster'] = clusters
print(loadings_df)
plt.figure(figsize=(8,6))
scatter = plt.scatter(loadings_df['PC1'], loadings_df['PC2'], c=loadings_df['Cluster'], cmap='viridis', s=100)
for country, row in loadings_df.iterrows():
    plt.text(row['PC1']+0.01, row['PC2']+0.01, country, fontsize=9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Country Clusters Based on PCA Loadings (2022)')
plt.colorbar(scatter, label='Cluster')
plt.show()

results = {}
for year, group in weekly_residuals_PCA.groupby('year'):
    group = group.drop(columns='year')
    group_std = (group - group.mean()) / group.std()
    pca = PCA(n_components=2)
    pca.fit(group_std)
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, index=group_std.columns, columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(loadings_df)
    loadings_df['Cluster'] = clusters
    results[year] = {
        'loadings': loadings_df[['PC1', 'PC2']],
        'explained_variance': pca.explained_variance_ratio_
    }
summary_data = []
for year, info in results.items():
    loadings_df = info['loadings']
    explained = info['explained_variance']
    for idx, comp in enumerate(loadings_df.columns):
        avg_loading = loadings_df[comp].mean()
        summary_data.append({
            'Year': year,
            'Component': comp,
            'Average_Loading': avg_loading,
            'Explained_Variance_Ratio': explained[idx]
        })
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by='Year').reset_index(drop=True)
print(summary_df)

# -------------------------------
# DCC Model
# -------------------------------
window = 60
rolling_vol = weekly_residuals_df.rolling(window=window).std()
std_resid = weekly_residuals_df / rolling_vol
std_resid = std_resid.dropna()
u = std_resid.values
T, N = u.shape

a_param = 0.02
b_param = 0.97
Q_bar = np.cov(u, rowvar=False)
Q_series = np.zeros((T, N, N))
R_series = np.zeros((T, N, N))
Q_prev = Q_bar.copy()
for t in range(T):
    if t == 0:
        Q_t = Q_bar.copy()
    else:
        outer_product = np.outer(u[t-1], u[t-1])
        Q_t = (1 - a_param - b_param) * Q_bar + a_param * outer_product + b_param * Q_prev
    Q_series[t] = Q_t
    diag_Q = np.sqrt(np.diag(Q_t))
    R_t = Q_t / np.outer(diag_Q, diag_Q)
    R_series[t] = R_t
    Q_prev = Q_t.copy()

avg_corr = []
for t in range(T):
    R_t = R_series[t]
    iu = np.triu_indices(N, k=1)
    avg_corr_t = np.mean(R_t[iu])
    avg_corr.append(avg_corr_t)
avg_corr = np.array(avg_corr)

plt.figure(figsize=(10, 6))
plt.plot(std_resid.index, avg_corr, label='Average Correlation', color='blue')
plt.xlabel('Tiempo')
plt.ylabel('Correlación promedio entra paises')
plt.title('Correlación promedio en el tiempo del modelo DCC')
plt.grid(True)
plt.show()

pair_corr = [R_series[t][0, 1] for t in range(T)]
plt.figure(figsize=(10, 6))
plt.plot(std_resid.index, pair_corr, label=f'Correlation: {data.columns[20]} & {data.columns[22]}', color='green')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title(f'Time-Varying Correlation between {data.columns[0]} and {data.columns[1]}')
plt.grid(True)
plt.show()
