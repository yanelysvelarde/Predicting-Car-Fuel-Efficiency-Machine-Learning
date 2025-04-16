from utils.timer import Timer as timer
with timer("LIBRARY/MODULE IMPORTING"):
    import os
    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.inspection import permutation_importance

def main():
    #Data to load
    df = pd.read_csv(os.path.normpath(os.path.join(os.path.dirname(__file__), "datasets", "1984-2026-vehicles.csv")), low_memory=False)

    with timer("DATASET PREP"):
        target = 'comb08'  #this is the label we’re predicting

        #this is to delete the rows missing the label value
        df.dropna(subset=[target], inplace=True)

        #y = MPG
        y = df[target]
        df = df.drop(columns=[target])  #remove label from features

        #remove that probly won’t help
        drop_cols = ['engId', 'make', 'model']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        #this is to delete the rows w any other missing values
        threshold = 0.5
        df = df.loc[:, df.isnull().mean() < threshold]
        df.dropna(inplace=True)

        #turn all categorical columns into numbers
        # this is gonna turn into several columns
        df = pd.get_dummies(df, drop_first=True)

        #X = all usable features
        X = df

        #80-20 training 
        X_train, X_test, y_train, y_test = train_test_split(X, y.loc[X.index], test_size=0.2, random_state=42)

        print(f"✅ using {X.shape[1]} features for training.")


    #3 models we are testing
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            max_features='sqrt', 
            bootstrap=True, 
            n_jobs=-1, 
            random_state=42
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=1000, 
            learning_rate=0.05, 
            max_depth=None, 
            l2_regularization=0.0, 
            early_stopping=False, 
            random_state=42
        )
    }

    results = {}
    with timer("MODEL TRAINING"):
        for name, model in models.items():
            with timer(f"{name.upper()} TRAINING"):
                print(f"Training: {name}...")
                model.fit(X_train, y_train)  #train the model
                preds = model.predict(X_test)  #predict MPG
                results[name] = {
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    "MAE": mean_absolute_error(y_test, preds),
                    "R² Score": r2_score(y_test, preds)
                }

    #print
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("\n", end="")

    #find best model:highest R² score
    best_model_name = max(results, key=lambda name: results[name]["R² Score"])

    #save all models;prefix best one:"BEST_"
    with timer("MODEL SAVING"):
        for name, model in models.items():
            filename_prefix = "BEST_" if name == best_model_name else ""
            model_filename = f"models/{filename_prefix}{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_filename)
            print(f"✅ Saved {name} → {model_filename}")


    #plot best model
    best_model = models[best_model_name]
    feature_names = X.columns
    
    if isinstance(best_model, HistGradientBoostingRegressor):
        X_small = X_test.sample(n=500, random_state=42)
        y_small = y_test.loc[X_small.index]
        result = permutation_importance(
            best_model, X_small, y_small, n_repeats=3, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean
    else:
        importances = best_model.feature_importances_

    top_n = 25
    sorted_idx = np.argsort(importances)[-top_n:]
    sorted_features = feature_names[sorted_idx]
    sorted_importances = importances[sorted_idx]

    #plot
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("BEST MODEL METRICS")
    plt.barh(sorted_features, sorted_importances)
    plt.title(f"Top {top_n} Feature Importances ({best_model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()