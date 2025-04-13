import joblib
import pandas as pd

#load saved model
model = joblib.load("random_forest_model.pkl")


input_data = pd.DataFrame([{
    'displ': 2.5,
    'cylinders': 4,
    'year': 2020,
    'drive_4-Wheel Drive': 0,
    'drive_All-Wheel Drive': 1,
    'drive_Front-Wheel Drive': 0,
    'drive_Part-time 4-Wheel Drive': 0,
    'drive_Rear-Wheel Drive': 0
}])

#predict MPG
predicted_mpg = model.predict(input_data)
print(f"Predicted MPG: {predicted_mpg[0]:.2f}")



