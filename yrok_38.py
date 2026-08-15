import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

df = pd.read_csv('text/covid_data.csv')

for c in ["new_cases", "new_deaths", "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

df["total_vaccinations"] = df["total_vaccinations"].fillna(0)
df = df.drop_duplicates()

# Sample data for faster processing
df_sample = df.sample(n=min(10000, len(df)), random_state=42)
df_sample['high_cases'] = (df_sample['new_cases'] > 1000).astype(int)
df_encoded = pd.get_dummies(df_sample, columns=['continent'], prefix='continent', dtype=int)
df_encoded['iso_code_encoded'] = LabelEncoder().fit_transform(df_encoded['iso_code'].astype(str))
df_encoded = df_encoded.sort_values(by=["location", "date"])
df_encoded['growth_rate_new_cases'] = df_encoded.groupby("location")["new_cases"].pct_change().fillna(0)
df_encoded.to_csv('text/covid_data_cleaned.csv', index=False)

cm = df_sample[["new_cases", "new_deaths", "total_cases", "population", "gdp_per_capita"]].corr()
print(cm)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=1, square=True, ax=axes[0, 0])
sns.boxplot(data=df_sample, x='continent', y='total_deaths_per_million', hue='continent', ax=axes[0, 1], legend=False)
sns.pairplot(df_sample[["total_cases", "total_deaths", "total_vaccinations", "population"]].sample(min(1000, len(df_sample))))
plt.suptitle('Pair Plot', y=1.02)
plt.savefig('fotog/urok_38_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

axes[1, 0].scatter(df_sample['total_cases'], df_sample['total_deaths'], alpha=0.5, color='teal')
df_sample[df_sample["date"] == df_sample["date"].max()].nlargest(10, "total_cases_per_million")[["location", "total_cases_per_million"]].plot(kind="bar", x="location", y="total_cases_per_million", ax=axes[1, 1])
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('fotog/urok_38_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Готово!")