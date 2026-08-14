import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

df = pd.read_csv('covid_data_cleaned.csv')
fc = ["total_cases", "total_deaths", "total_vaccinations", "population", "gdp_per_capita", "iso_code_encoded"]
fc.extend([c for c in df.columns if c.startswith('continent_')])
X = df[fc].copy()
y_cls, y_reg = df['high_cases'], df['new_cases']

scaler = StandardScaler()
X[["total_cases", "total_deaths", "total_vaccinations"]] = scaler.fit_transform(X[["total_cases", "total_deaths", "total_vaccinations"]])

Xtrc, Xtec, ytrc, ytec = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# Крос-валідація
models = {
    "RF": RandomForestClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42)
}

for n, m in models.items():
    s = cross_val_score(m, Xtrc, ytrc, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="f1")
    print(f"{n}: F1={s.mean():.4f}, Std={s.std():.4f}")

# GridSearch
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
    cv=3,
    scoring="f1"
)
grid.fit(Xtrc, ytrc)
print(f"Best: {grid.best_params_}, F1={grid.best_score_:.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42).fit(Xtrc, ytrc)
print(f"GB F1: {f1_score(ytec, gb.predict(Xtec), zero_division=0):.4f}")

# Feature Importance
rf = RandomForestClassifier(random_state=42).fit(Xtrc, ytrc)
idx = np.argsort(rf.feature_importances_)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_[idx])
plt.xticks(range(len(rf.feature_importances_)), [X.columns[i] for i in idx], rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('fotog/urok_40_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

for i, ix in enumerate(idx[:5]):
    print(f"{i+1}. {X.columns[ix]}: {rf.feature_importances_[ix]:.4f}")

# ROC
plt.figure(figsize=(10, 8))
for n, m in models.items():
    m.fit(Xtrc, ytrc)
    fpr, tpr, _ = roc_curve(ytec, m.predict_proba(Xtec)[:, 1])
    plt.plot(fpr, tpr, label=f'{n} (AUC={auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.legend()
plt.savefig('fotog/urok_40_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# SelectKBest
k = min(10, Xtrc.shape[1])
skb = SelectKBest(score_func=f_classif, k=k)
Xtrk, Xtek = skb.fit_transform(Xtrc, ytrc), skb.transform(Xtec)
grid.best_estimator_.fit(Xtrk, ytrc)
print(f"Top-{k} F1: {f1_score(ytec, grid.best_estimator_.predict(Xtek), zero_division=0):.4f}")
print("Готово!")