import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('text/covid_data.csv')
for c in ["new_cases", "new_deaths", "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million"]:
    if c in df.columns: df[c] = df[c].fillna(df[c].median())
df["total_vaccinations"] = df["total_vaccinations"].fillna(0)
df = df.drop_duplicates()
df_sample = df.sample(n=min(10000, len(df)), random_state=42)
print(df_sample[["new_cases", "new_deaths", "total_cases", "total_deaths", "population", "gdp_per_capita"]].describe())
cnt = ["Ukraine", "United States", "Germany"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for c in cnt:
    cd = df[df["location"] == c]
    axes[0, 0].plot(cd["date"], cd["total_cases"], label=c, linewidth=2)
    axes[0, 1].plot(cd["date"], cd["total_deaths"], label=c, linewidth=2)
axes[0, 0].set_title('Загальні випадки'); axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 1].set_title('Загальні смерті'); axes[0, 1].tick_params(axis='x', rotation=45)
df[df["date"] == df["date"].max()].groupby("continent")["total_cases"].sum().plot(kind="bar", ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Випадки за континентами'); axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 1].hist(df_sample["total_cases"], bins=30, alpha=0.7, label='Випадки', color='blue')
axes[1, 1].hist(df_sample["total_deaths"], bins=30, alpha=0.7, label='Смерті', color='red')
axes[1, 1].set_title('Розподіл'); axes[1, 1].legend()
plt.tight_layout(); plt.savefig('fotog/urok_37_analysis.png', dpi=300, bbox_inches='tight'); plt.close()
print("Готово!")