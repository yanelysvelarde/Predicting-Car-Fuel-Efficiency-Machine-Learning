import os
import joblib
import pandas as pd

#load the model
model = joblib.load(os.path.normpath(os.path.join(os.path.dirname(__file__), "models", "BEST_gradient_boosting.pkl")))

#load and preprocess original dataset -- feature columns
df = pd.read_csv(os.path.normpath(os.path.join(os.path.dirname(__file__), "datasets", "1984-2026-vehicles.csv")), low_memory=False)
df.dropna(subset=['comb08'], inplace=True)
df.drop(columns=['comb08'], inplace=True)

drop_cols = ['engId', 'make', 'model']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

threshold = 0.5
df = df.loc[:, df.isnull().mean() < threshold]
df.dropna(inplace=True)

#one-hot encode dataset to get full feature set
df_encoded = pd.get_dummies(df, drop_first=True)
full_feature_list = df_encoded.columns

#create custom input
custom_input = {
    'displ': 2.5,
    'cylinders': 4,
    'year': 2020,
    'drive_All-Wheel Drive': 1,
    'drive_4-Wheel Drive': 0,
    'drive_Front-Wheel Drive': 0,
    'drive_Part-time 4-Wheel Drive': 0,
    'drive_Rear-Wheel Drive': 0
}

#build DataFrame and fill missing columns with 0
input_data = pd.DataFrame([custom_input])

#build missing columns in one go
missing_cols = [col for col in full_feature_list if col not in input_data.columns]
missing_df = pd.DataFrame([{col: 0 for col in missing_cols}])

#combine original input and missing columns
input_data = pd.concat([input_data, missing_df], axis=1)

#reorder columns to match model training
input_data = input_data[full_feature_list]

#predict
predicted_mpg = model.predict(input_data)
print(f"Predicted MPG: {predicted_mpg[0]:.2f}")
