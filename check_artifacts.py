from pathlib import Path
print("model:    ", Path("model/churn_model.pkl").exists())
print("explainer:", Path("model/shap_explainer.pkl").exists())
print("features: ", Path("data/feature_columns.json").exists())
print("csv:      ", Path("data/telco_churn.csv").exists())
print("train:    ", Path("data/train.parquet").exists())
print("test:     ", Path("data/test.parquet").exists())
