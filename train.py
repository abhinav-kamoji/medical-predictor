import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Load the dataset (matching your exact file name)
df = pd.read_csv('medical-charges.csv')

# 2. Preprocess Data
# This automatically converts 'sex', 'smoker', and 'region' into 0 and 1 columns.
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model (Optional, but good for your CV)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(f"Model R^2 Score: {score:.2f} (Explains {score*100:.1f}% of the variance)")

# 6. SAVE THE MODEL as model.pkl
joblib.dump(model, 'model.pkl')

print("Success! model.pkl has been created in your folder.")