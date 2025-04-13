import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Data to load
df = pd.read_csv("1984-2026-vehicles.csv", low_memory=False)


features = ['displ', 'cylinders', 'drive', 'year']
target = 'comb08'
df_model = df[features + [target]].copy()

#this is to delete the rows w no values
df_model.dropna(inplace=True)

#turn 'drive' column into numbers
# this is gonna turn into several columns
df_model = pd.get_dummies(df_model, columns=['drive'], drop_first=True)

#y = MPG
X = df_model.drop(columns=[target])
y = df_model[target]

#80-20 training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3 models we are testing
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  #train the model
    preds = model.predict(X_test)  #predict MPG
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R² Score": r2_score(y_test, preds)
    }

#print
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

#save the best performing model
joblib.dump(models["Random Forest"], "random_forest_model.pkl")

importances = models["Random Forest"].feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

