import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=100, freq="D")
countries = [
    ("Ukraine", "UKR", "Europe", 43000000, 13000),
    ("United States", "USA", "North America", 331000000, 63000),
    ("Germany", "DEU", "Europe", 83000000, 51000),
    ("India", "IND", "Asia", 1380000000, 21000),
    ("Brazil", "BRA", "South America", 212000000, 14000),
    ("South Africa", "ZAF", "Africa", 59000000, 13000)
]

data = []
for country, iso, continent, pop, gdp in countries:
    c_cases, c_deaths, c_vac = 0, 0, 0
    for i, date in enumerate(dates):
        n_cases = int(np.random.poisson(500 * (1 + 0.05 * i)))
        n_deaths = int(n_cases * np.random.uniform(0.01, 0.03))
        c_cases += n_cases
        c_deaths += n_deaths
        c_vac += int(np.random.poisson(1000 * (1 if i > 30 else 0)))
        
        data.append({
            "iso_code": iso,
            "continent": continent,
            "location": country,
            "date": date.strftime("%Y-%m-%d"),
            "total_cases": float(c_cases),
            "new_cases": float(n_cases),
            "total_deaths": float(c_deaths),
            "new_deaths": float(n_deaths),
            "total_cases_per_million": float((c_cases / pop) * 1e6),
            "total_deaths_per_million": float((c_deaths / pop) * 1e6),
            "total_vaccinations": float(c_vac),
            "population": float(pop),
            "gdp_per_capita": float(gdp)
        })

df = pd.DataFrame(data)

for col, rs in [("new_cases", 42), ("new_deaths", 24), ("total_cases", 10)]:
    df.loc[df.sample(frac=0.03, random_state=rs).index, col] = np.nan

df.to_csv("owid-covid-data.csv", index=False)