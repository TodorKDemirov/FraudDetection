import joblib

model = joblib.load("../models/xgboost_final.joblib")

print("Expected features:")
for i, feature in enumerate(model.feature_names_in_):
    print(i, feature)