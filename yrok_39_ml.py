import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

df = pd.read_csv('text/covid_data_cleaned.csv')

# Sample data for faster processing
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

fc = ["total_cases", "total_deaths", "total_vaccinations", "population", "gdp_per_capita", "iso_code_encoded"]
fc.extend([c for c in df_sample.columns if c.startswith('continent_')])
X = df_sample[fc].copy()

# Remove NaN values
X = X.dropna()
df_sample_clean = df_sample.loc[X.index]
y_cls, y_reg = df_sample_clean['high_cases'], df_sample_clean['new_cases']

scaler = StandardScaler()
X[["total_cases", "total_deaths", "total_vaccinations"]] = scaler.fit_transform(X[["total_cases", "total_deaths", "total_vaccinations"]])

Xtrc, Xtec, ytrc, ytec = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
Xtrr, Xter, ytrr, yter = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Класифікація
cls_models = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "DT": DecisionTreeClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "kNN": KNeighborsClassifier()
}

cls_res = []
for n, m in cls_models.items():
    m.fit(Xtrc, ytrc)
    p = m.predict(Xtec)
    cls_res.append([n, accuracy_score(ytec, p), precision_score(ytec, p, zero_division=0), recall_score(ytec, p, zero_division=0), f1_score(ytec, p, zero_division=0)])

print(pd.DataFrame(cls_res, columns=["Model", "Acc", "Prec", "Rec", "F1"]).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for i, (n, m) in enumerate(cls_models.items()):
    m.fit(Xtrc, ytrc)
    sns.heatmap(confusion_matrix(ytec, m.predict(Xtec)), annot=True, fmt='d', cmap='Blues', ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(n)

plt.tight_layout()
plt.savefig('fotog/urok_39_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Регресія
reg_models = {
    "Linear": LinearRegression(),
    "Poly": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Ridge": Ridge(alpha=10.0, random_state=42, solver='auto'),
    "Lasso": Lasso(alpha=0.1, max_iter=5000, random_state=42)
}

reg_res = []
for n, m in reg_models.items():
    m.fit(Xtrr, ytrr)
    p = m.predict(Xter)
    reg_res.append([n, mean_squared_error(yter, p), np.sqrt(mean_squared_error(yter, p)), mean_absolute_error(yter, p), r2_score(yter, p)])

print(pd.DataFrame(reg_res, columns=["Model", "MSE", "RMSE", "MAE", "R2"]).to_string(index=False))

# Графіки регресії
br = Ridge(alpha=10.0, random_state=42, solver='auto').fit(Xtrr, ytrr)
pr = br.predict(Xter)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(yter, pr, alpha=0.5, color='blue')
axes[0].plot([yter.min(), yter.max()], [yter.min(), yter.max()], 'r--', lw=2)
axes[0].set_title('Predicted vs Actual')
axes[1].hist(yter - pr, bins=30, color='green', alpha=0.7)
axes[1].set_title('Residuals')

plt.tight_layout()
plt.savefig('fotog/urok_39_regression_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Готово!")