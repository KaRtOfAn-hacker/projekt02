import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load data
df = pd.read_csv('text/covid_data.csv')

# Data preprocessing (matching main.py logic)
for col in ["new_cases", "new_deaths", "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
df["total_vaccinations"] = df["total_vaccinations"].fillna(0)
df = df.drop_duplicates()

# Create sample for analysis
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

# Chart 1: Country trends (for Slide 9)
selected_countries = ["Ukraine", "United States", "Germany"]
plt.figure(figsize=(12, 6))
for country in selected_countries:
    cd = df[df["location"] == country]
    plt.plot(cd["date"], cd["total_cases"], label=country, linewidth=2)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Загальні випадки", fontsize=12)
plt.title("Динаміка захворюваності за країнами", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fotog/chart_country_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: Continent comparison (for Slide 9)
latest_data = df[df["date"] == df["date"].max()]
plt.figure(figsize=(10, 6))
latest_data.groupby("continent")["total_cases"].sum().plot(kind="bar", color='steelblue')
plt.xlabel("Континент", fontsize=12)
plt.ylabel("Загальні випадки", fontsize=12)
plt.title("Розподіл випадків за континентами", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fotog/chart_continent_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: Correlation heatmap (for Slide 11)
correlation_matrix = df_sample[["new_cases", "new_deaths", "total_cases", "population", "gdp_per_capita"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', linewidths=1, square=True)
plt.title("Кореляційна матриця ключових показників", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fotog/chart_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 4: Feature distribution (for Slide 9)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_sample["total_cases"], bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel("Загальні випадки", fontsize=12)
axes[0].set_ylabel("Частота", fontsize=12)
axes[0].set_title("Розподіл загальних випадків", fontsize=12, fontweight='bold')

axes[1].hist(df_sample["total_deaths"], bins=30, alpha=0.7, color='red', edgecolor='black')
axes[1].set_xlabel("Загальні смерті", fontsize=12)
axes[1].set_ylabel("Частота", fontsize=12)
axes[1].set_title("Розподіл загальних смертей", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fotog/chart_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 5: Model comparison table (create as image for Slide 10)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# Prepare data for ML
df_sample["iso_code_encoded"] = LabelEncoder().fit_transform(df_sample["iso_code"].astype(str))
df_sample["high_cases"] = (df_sample["new_cases"] > 1000).astype(int)
X = pd.concat([df_sample[["total_cases", "total_deaths", "total_vaccinations", "population", "gdp_per_capita", "iso_code_encoded"]].copy(), 
               pd.get_dummies(df_sample["continent"], prefix="continent")], axis=1)
X = X.dropna()
df_clean = df_sample.loc[X.index]
y_reg, y_cls = df_clean["new_cases"].values, df_clean["high_cases"].values

scaler = StandardScaler()
X[["total_cases", "total_deaths", "total_vaccinations"]] = scaler.fit_transform(X[["total_cases", "total_deaths", "total_vaccinations"]])

X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(X, y_reg, y_cls, test_size=0.2, random_state=42)

# Classification models
cls_models = {
    "Logistic Regression": make_pipeline(SMOTE(random_state=42), LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(SMOTE(random_state=42), DecisionTreeClassifier(random_state=42)),
    "Random Forest": make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=42)),
    "k-NN": make_pipeline(SMOTE(random_state=42), KNeighborsClassifier()),
    "Gradient Boosting": make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42))
}

cls_results = []
for name, model in cls_models.items():
    model.fit(X_train, y_train_cls)
    y_pred = model.predict(X_test)
    cls_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_cls, y_pred),
        "Precision": precision_score(y_test_cls, y_pred, zero_division=0),
        "Recall": recall_score(y_test_cls, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test_cls, y_pred, zero_division=0)
    })

# Create model comparison chart
cls_df = pd.DataFrame(cls_results)
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=cls_df.round(3).values, colLabels=cls_df.columns, 
                 cellLoc='center', loc='center', colColours=['#1e3a8a']*5)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
for i in range(len(cls_df.columns)):
    table[(0, i)].set_facecolor('#1e3a8a')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title("Порівняння моделей класифікації", fontsize=14, fontweight='bold', pad=20)
plt.savefig('fotog/chart_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 6: Feature importance (for Slide 11)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train_cls)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel("Важливість ознаки", fontsize=12)
plt.ylabel("Ознака", fontsize=12)
plt.title("Топ-10 найважливіших ознак (Random Forest)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fotog/chart_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 7: Boxplot by continent (for Slide 9)
plt.figure(figsize=(10, 6))
sns.boxplot(x="continent", y="total_deaths_per_million", data=df_sample)
plt.xlabel("Континент", fontsize=12)
plt.ylabel("Смертей на мільйон", fontsize=12)
plt.title("Розподіл смертей на мільйон за континентами", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fotog/chart_boxplot_continents.png', dpi=300, bbox_inches='tight')
plt.close()

print("All charts generated successfully!")
print("Charts saved to fotog/ directory")
