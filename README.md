# Predicting-Car-Fuel-Efficiency-Machine-Learning
Predicting Car Fuel Efficiency Based on Vehicle Attributes

Contributors: Ernesto Leiva, Osasenaga Imafidon, Maria Yanelys Velarde.

Objective:
This project aims to predict a vehicle’s fuel efficiency (measured in miles per gallon, MPG) based on various car attributes such as engine displacement, horsepower, weight, number of cylinders, model year, and drivetrain type. With rising fuel costs and growing emphasis on sustainability, understanding the factors that affect fuel consumption is both timely and valuable.

Motivation:
Accurately predicting fuel efficiency is critical for consumers, manufacturers, and policymakers. By leveraging machine learning models on historical and modern car data, we can explore how vehicle design choices impact MPG and identify trends over time.

Datasets:

We plan to use the following sources:

Auto MPG Dataset (UCI Machine Learning Repository)
Link: https://archive.ics.uci.edu/dataset/9/auto+mpg
This dataset contains information on cars from the late 1970s and early 1980s.

Fuel Economy Data (U.S. Department of Energy)
Link: https://www.fueleconomy.gov/feg/download.shtml
This dataset provides detailed fuel economy and emissions data for vehicles manufactured from 1984 through 2025.

Note: If the attribute formats or units between these two datasets are too inconsistent to reasonably merge, we will default to using only the 1984–2025 dataset, which is larger and more comprehensive.

Methodology:

Data Cleaning & Preprocessing: Normalize and merge datasets if possible. Handle missing or inconsistent values. If merging proves infeasible, we will proceed with only the modern dataset (1984–2025).
Exploratory Data Analysis: Visualize trends and correlations between variables and MPG.
Feature Engineering: Create new relevant features such as power-to-weight ratio or engine efficiency metrics.
Modeling: Train and evaluate multiple regression models (e.g., Linear Regression, Random Forest, Gradient Boosting, and potentially a Neural Network).
Evaluation Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.

Deliverables:

A cleaned and combined dataset spanning vehicles from the 1970s to 2025.
Comparative analysis of model performance.
A final trained model capable of predicting MPG based on vehicle specs.
Visualizations showing trends in efficiency across decades and feature importance.
A final report and presentation summarizing our findings and conclusions.
