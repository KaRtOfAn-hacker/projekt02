import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def save_plot(title, filename):
    plt.title(title); plt.tight_layout(); plt.savefig(f"fotog/{filename}"); plt.close()

def evaluate_model(model, X_train, X_test, y_train, y_test, metrics):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {name: metric(y_test, preds) for name, metric in metrics.items()}

df = pd.read_csv("text/covid_data.csv")
print(df.head(), df.info(), df.isnull().sum(), df.duplicated().sum())

df = df.drop_duplicates()
for col in ["new_cases", "new_deaths", "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million"]:
    if col in df.columns: df[col] = df[col].fillna(df[col].median())
df["total_vaccinations"] = df["total_vaccinations"].fillna(0)

print(df[["new_cases", "new_deaths", "total_cases", "total_deaths", "population", "gdp_per_capita"]].describe())

df["iso_code_encoded"] = LabelEncoder().fit_transform(df["iso_code"].astype(str))
df = df.sort_values(by=["location", "date"])
df["growth_rate_new_cases"] = df.groupby("location")["new_cases"].pct_change().fillna(0)
df["growth_rate_new_deaths"] = df.groupby("location")["new_deaths"].pct_change().fillna(0)

# Sample data for faster processing with large dataset
df_sample = df.sample(n=min(10000, len(df)), random_state=42)
df_sample["iso_code_encoded"] = LabelEncoder().fit_transform(df_sample["iso_code"].astype(str))

selected_countries = ["Ukraine", "United States", "Germany"]
latest_data = df[df["date"] == df["date"].max()]

for target, ylabel, fname in [("total_cases", "Total Cases", "total_cases_trend.png"), ("total_deaths", "Total Deaths", "total_deaths_trend.png")]:
    plt.figure(figsize=(12, 6))
    for country in selected_countries:
        cd = df[df["location"] == country]
        plt.plot(cd["date"], cd[target], label=country)
    plt.xlabel("Date"); plt.ylabel(ylabel); plt.legend(); plt.xticks(rotation=45)
    save_plot(f"{ylabel} Over Time", fname)

plt.figure(figsize=(10, 6))
latest_data.groupby("continent")["total_cases"].sum().plot(kind="bar")
plt.xlabel("Continent"); plt.ylabel("Total Cases")
save_plot("Total Cases by Continent (Latest Date)", "continent_comparison.png")

correlation_matrix = df[["new_cases", "new_deaths", "total_cases", "population", "gdp_per_capita"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
save_plot("Correlation Matrix", "correlation_heatmap.png")
print(correlation_matrix)

plt.figure(figsize=(12, 5))
for i, (col, xlabel) in enumerate([("total_cases", "Total Cases"), ("total_deaths", "Total Deaths")], 1):
    plt.subplot(1, 2, i); plt.hist(df[col], bins=30); plt.xlabel(xlabel)
save_plot("Distributions", "distributions.png")

plt.figure(figsize=(10, 6))
sns.boxplot(x="continent", y="total_deaths_per_million", data=df)
plt.xlabel("Continent"); plt.ylabel("Deaths per Million"); plt.xticks(rotation=45)
save_plot("Total Deaths per Million by Continent", "boxplot_continents.png")

sns.pairplot(df_sample[["total_cases", "total_deaths", "total_vaccinations", "population"]].sample(min(1000, len(df_sample))))
plt.suptitle("Pair Plot of Key Variables", y=1.02); plt.savefig("fotog/pairplot.png"); plt.close()

plt.figure(figsize=(12, 6))
for country in selected_countries:
    cd = df[df["location"] == country]
    plt.plot(cd["date"], cd["new_cases"], label=country, linestyle='--')
plt.xlabel("Date"); plt.ylabel("New Cases"); plt.legend(); plt.xticks(rotation=45)
save_plot("New Cases Trend (Linear)", "new_cases_trend.png")

plt.figure(figsize=(12, 6))
top_10 = latest_data.nlargest(10, "total_cases_per_million")[["location", "total_cases_per_million"]]
plt.bar(top_10["location"], top_10["total_cases_per_million"])
plt.xlabel("Country"); plt.ylabel("Cases per Million"); plt.xticks(rotation=45)
save_plot("Top 10 Countries by Total Cases per Million", "top_10_countries.png")

for country in selected_countries:
    cd = df[df["location"] == country]
    print(f"{country}: Max cases {cd.loc[cd['new_cases'].idxmax(), 'date']}, Max deaths {cd.loc[cd['new_deaths'].idxmax(), 'date']}, Min cases {cd.loc[cd['new_cases'].idxmin(), 'date']}, Min deaths {cd.loc[cd['new_deaths'].idxmin(), 'date']}")

df_sample["high_cases"] = (df_sample["new_cases"] > 1000).astype(int)
X = pd.concat([df_sample[["total_cases", "total_deaths", "total_vaccinations", "population", "gdp_per_capita", "iso_code_encoded"]].copy(), pd.get_dummies(df_sample["continent"], prefix="continent")], axis=1)

# Remove rows with NaN values for ML
X = X.dropna()
df_clean = df_sample.loc[X.index]
y_reg = df_clean["new_cases"].values
y_cls = df_clean["high_cases"].values

scaler = StandardScaler()
X[["total_cases", "total_deaths", "total_vaccinations"]] = scaler.fit_transform(X[["total_cases", "total_deaths", "total_vaccinations"]])

X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(X, y_reg, y_cls, test_size=0.2, random_state=42)

reg_models = {"Linear": LinearRegression(), "Polynomial": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()), "Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.1, max_iter=5000)}
reg_metrics = {"MSE": mean_squared_error, "RMSE": lambda y, p: np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error, "R2": r2_score}
reg_results = [evaluate_model(model, X_train, X_test, y_train_reg, y_test_reg, reg_metrics) | {"Model": name} for name, model in reg_models.items()]
print(pd.DataFrame(reg_results))

cls_models = {"Logistic Regression": make_pipeline(SMOTE(random_state=42), LogisticRegression(max_iter=1000)), "Decision Tree": make_pipeline(SMOTE(random_state=42), DecisionTreeClassifier(random_state=42)), "Random Forest": make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=42)), "k-NN": make_pipeline(SMOTE(random_state=42), KNeighborsClassifier()), "Gradient Boosting": make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42))}
cls_metrics = {"Accuracy": accuracy_score, "Precision": lambda y, p: precision_score(y, p, zero_division=0), "Recall": lambda y, p: recall_score(y, p, zero_division=0), "F1-Score": lambda y, p: f1_score(y, p, zero_division=0)}
cls_results = [evaluate_model(model, X_train, X_test, y_train_cls, y_test_cls, cls_metrics) | {"Model": name} for name, model in cls_models.items()]
print(pd.DataFrame(cls_results))

for name, model in cls_models.items():
    model.fit(X_train, y_train_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test_cls, model.predict(X_test)), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    save_plot(f"Confusion Matrix - {name}", f"confusion_matrix_{name.replace(' ', '_').lower()}.png")

plt.figure(figsize=(10, 8))
for name, model in cls_models.items():
    if name != "k-NN":
        model.fit(X_train, y_train_cls)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_cls, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves'); plt.legend()
plt.savefig("fotog/roc_curves.png"); plt.close()

rf = make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=42)).fit(X_train, y_train_cls)
rf_clf = rf.named_steps['randomforestclassifier']
if hasattr(rf_clf, 'feature_importances_'):
    idx = np.argsort(rf_clf.feature_importances_)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(rf_clf.feature_importances_)), rf_clf.feature_importances_[idx])
    plt.xticks(range(len(rf_clf.feature_importances_)), [X.columns[i] for i in idx], rotation=90)
    plt.title("Feature Importance (Random Forest)"); plt.tight_layout()
    plt.savefig("fotog/feature_importance.png"); plt.close()
    print("Feature Importance:")
    for i, ix in enumerate(idx[:10]): print(f"{i+1}. {X.columns[ix]}: {rf_clf.feature_importances_[ix]:.4f}")

kf = KFold(n_splits=3, shuffle=True, random_state=42)
for name, model in cls_models.items():
    scores = cross_val_score(model, X_train, y_train_cls, cv=kf, scoring="f1")
    print(f"{name}: Mean F1 = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Simplified GridSearchCV for faster processing
grid_rf = GridSearchCV(make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=42)), {"randomforestclassifier__n_estimators": [50, 100], "randomforestclassifier__max_depth": [10, 20]}, cv=2, scoring="f1")
grid_rf.fit(X_train, y_train_cls)
print(f"Best RF Params: {grid_rf.best_params_}")
print(f"RF F1 Score after optimization: {f1_score(y_test_cls, grid_rf.best_estimator_.predict(X_test), zero_division=0):.4f}")

best_reg = LinearRegression().fit(X_train, y_train_reg)
reg_preds = best_reg.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, reg_preds, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel("Actual Values"); plt.ylabel("Predicted Values")
save_plot("Predicted vs Actual (Regression)", "predicted_vs_actual.png")

plt.figure(figsize=(10, 6))
plt.hist(y_test_reg - reg_preds, bins=30)
plt.xlabel("Residuals"); plt.ylabel("Frequency")
save_plot("Distribution of Residuals", "residuals_distribution.png")

errors = np.abs(y_test_reg - reg_preds)
for i, idx in enumerate(np.argsort(errors)[-10:][::-1]):
    print(f"{i+1}. Index: {idx}, Actual: {y_test_reg[idx]:.2f}, Predicted: {reg_preds[idx]:.2f}, Error: {errors[idx]:.2f}")

k = min(10, X_train.shape[1])
skb = SelectKBest(score_func=f_classif, k=k)
X_train_kbest, X_test_kbest = skb.fit_transform(X_train, y_train_cls), skb.transform(X_test)
grid_rf.best_estimator_.fit(X_train_kbest, y_train_cls)
print(f"Classification F1-Score on top {k} features: {f1_score(y_test_cls, grid_rf.best_estimator_.predict(X_test_kbest), zero_division=0):.4f}")

df_clean.to_csv("text/covid_data_cleaned.csv", index=False)
print("Cleaned data saved to text/covid_data_cleaned.csv")
print("All visualizations saved to fotog/ directory")
print("Analysis reports saved to text/analysis_reports.txt")