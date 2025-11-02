import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]

df = pd.read_csv(url, names=columns, na_values="?")
df = df.dropna().reset_index(drop=True)
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])
df.to_csv("data/heart.csv", index=False)

print("Saved heart.csv with shape:", df.shape)
print(df.head(10))
