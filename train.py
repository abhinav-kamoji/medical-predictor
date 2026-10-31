from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import joblib

df = pd.read_csv('medical-charges.csv')
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
score = r2_score(y_test, predictions)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Optimized Model R^2 Score: {score:.2f}")

joblib.dump(best_model, 'model.pkl')
print("Success! Optimized model.pkl has been created.")
