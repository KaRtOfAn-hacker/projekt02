import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

df = pd.read_csv("owid-covid-data.csv")

df["iso_code_encoded"] = LabelEncoder().fit_transform(df["iso_code"].astype(str))
df = df.sort_values(by=["location", "date"])
df["growth_rate_new_cases"] = df.groupby("location")["new_cases"].pct_change().fillna(0)
df["growth_rate_new_deaths"] = df.groupby("location")["new_deaths"].pct_change().fillna(0)

df["total_vaccinations"] = df["total_vaccinations"].fillna(0)

for col in ["new_cases", "new_deaths", "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates()

df["high_cases"] = (df["new_cases"] > 1000).astype(int)

X = df[["total_cases", "total_deaths", "total_vaccinations", "population", "gdp_per_capita", "iso_code_encoded"]].copy()
ohe_continent = pd.get_dummies(df["continent"], prefix="continent")
X = pd.concat([X, ohe_continent], axis=1)

scaler = StandardScaler()
num_features = ["total_cases", "total_deaths", "total_vaccinations"]
X[num_features] = scaler.fit_transform(X[num_features])

y_reg = df["new_cases"].values
y_cls = df["high_cases"].values

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_cls_res, y_train_cls_res = smote.fit_resample(X_train_cls, y_train_cls)

reg_models = {
    "Linear": LinearRegression(),
    "Polynomial": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=5000)
}

reg_results = []
for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    preds = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, preds)
    r2 = r2_score(y_test_reg, preds)
    reg_results.append({"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})

cls_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "k-NN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

cls_results = []
for name, model in cls_models.items():
    model.fit(X_train_cls_res, y_train_cls_res)
    preds = model.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, preds)
    prec = precision_score(y_test_cls, preds, zero_division=0)
    rec = recall_score(y_test_cls, preds, zero_division=0)
    f1 = f1_score(y_test_cls, preds, zero_division=0)
    cm = confusion_matrix(y_test_cls, preds)
    fn = cm[1][0] if cm.shape == (2, 2) else 0
    cls_results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "FN": fn})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in cls_models.items():
    scores = cross_val_score(model, X_train_cls_res, y_train_cls_res, cv=kf, scoring="f1")
    print(f"{name}: Mean F1 = {scores.mean():.4f}, Std = {scores.std():.4f}")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="f1")
grid_rf.fit(X_train_cls_res, y_train_cls_res)
best_rf = grid_rf.best_estimator_

rf_f1_before = cls_results[2]["F1-Score"]
rf_f1_after = f1_score(y_test_cls, best_rf.predict(X_test_cls), zero_division=0)
print(f"Best RF Params: {grid_rf.best_params_}")
print(f"RF F1 Before Optimization: {rf_f1_before:.4f} | After: {rf_f1_after:.4f}")

best_reg = LinearRegression()
best_reg.fit(X_train_reg, y_train_reg)
reg_preds = best_reg.predict(X_test_reg)

residuals = y_test_reg - reg_preds

k = min(10, X_train_cls_res.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_kbest = selector.fit_transform(X_train_cls_res, y_train_cls_res)
X_test_kbest = selector.transform(X_test_cls)

best_rf.fit(X_train_kbest, y_train_cls_res)
kbest_preds = best_rf.predict(X_test_kbest)
print(f"Classification F1-Score on top {k} features: {f1_score(y_test_cls, kbest_preds, zero_division=0):.4f}")