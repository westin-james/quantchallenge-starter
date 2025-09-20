import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .visualize import plot_all

######################## 1. LOADING DATA ########################
print("\n1. LOADING DATA\n")
print("Loading train.csv:")
column_names = ['time','A','B','C','D','E','F','G','H','I','J','K','L','M','N','Y1','Y2']

try:
    train_data = pd.read_csv('./data/train.csv',skiprows=1,names=column_names)
    print("Loaded train.csv")
except Exception as e:
    print(f"Error loading train.csv\n{e}")
    exit()

print("Loading test.csv")
test_column_names = ['id','time','A','B','C','D','E','F','G','H','I','J','K','L','M','N']

try:
    test_data = pd.read_csv('./data/test.csv',skiprows=1,names=['id']+column_names[:-2])
    print("Loaded test.csv")
except Exception as e:
    print(f"Error loading test.csv\n{e}")
    exit()

feature_cols = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
X_train = train_data[feature_cols]
y1_train = train_data['Y1']
y2_train = train_data['Y2']
X_test = test_data[feature_cols]

######################## 2. CROSS VALIDATION SETUP ########################
print("\n\n2. CROSS VALIDATION SETUP\n")
tscv = TimeSeriesSplit(n_splits=3)
print("Using 3-fold time series crossval")

######################## 3. LINEAR MODELS ########################
print("\n\n3. LINEAR MODEL BASELINE\n")

linear_y1 = LinearRegression()
linear_y2 = Ridge(alpha=100.0)

y1_linear_cv = cross_val_score(linear_y1, train_data[feature_cols], y1_train, cv=tscv, scoring='r2')

scaler = StandardScaler()
X_y2_scaled = scaler.fit_transform(train_data[feature_cols])
y2_linear_cv = cross_val_score(linear_y2, X_y2_scaled, y2_train, cv=tscv, scoring='r2')

print(f"Linear Y1 CV R^2: {y1_linear_cv.mean():.4f} (+/- {y1_linear_cv.std()*2:.4f})")
print(f"Linear Y2 CV R^2: {y2_linear_cv.mean():.4f} (+/- {y2_linear_cv.std()*2:.4f})")

######################## 4. RANDOM FOREST ########################
print("\n\n4. RANDOM FOREST\n")

rf_y1 = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_y2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

y1_rf_cv = cross_val_score(rf_y1, X_train, y1_train, cv=tscv, scoring='r2')
y2_rf_cv = cross_val_score(rf_y2, X_train, y2_train, cv=tscv, scoring='r2')

print(f"Random Forest Y1 CV R^2: {y1_rf_cv.mean():.4f} (+/- {y1_rf_cv.std()*2:.4f})")
print(f"Random Forest Y2 CV R^2: {y2_rf_cv.mean():.4f} (+/- {y2_rf_cv.std()*2:.4f})")

######################## 5. MODEL COMPARISON ########################
print("\n\n5. MODEL COMPARISON\n")

results = pd.DataFrame({
    'Model': ['Linear', 'Random Forest'],
    'Y1_R^2': [y1_linear_cv.mean(), y1_rf_cv.mean()],
    'Y2_R^2': [y2_linear_cv.mean(), y2_rf_cv.mean()]
})

results['Combined'] = (results['Y1_R^2'] + results['Y2_R^2']) / 2
results = results.sort_values('Combined', ascending=False)

print(results.to_string(index=False))

######################## 6. TRAIN FINAL MODELS ########################
print("\n\n6. TRAIN FINAL MODELS\n")

linear_y1.fit(train_data[feature_cols], y1_train)
linear_y2.fit(X_y2_scaled, y2_train)
rf_y1.fit(X_train, y1_train)
rf_y2.fit(X_train, y2_train)

######################## 7. FEATURE IMPORTANCE ########################
print("\n\n7. FEATURE IMPORTANCE\n")

rf_y1_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_y1.feature_importances_
}).sort_values('importance', ascending=False)

rf_y2_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_y2.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 features for Y1 (Random Forest):")
print(rf_y1_importance.head(5).to_string(index=False))
print("\nTop 5 features for Y2 (Random Forest):")
print(rf_y2_importance.head(5).to_string(index=False))

######################## 8. PREDICITIONS ########################
print("\n\n8. PREDICITIONS\n")

best_y1_model = results.loc[results['Y1_R^2'].idxmax(), 'Model']
best_y2_model = results.loc[results['Y2_R^2'].idxmax(), 'Model']

print(f"Best model for Y1: {best_y1_model}")
print(f"Best model for Y2: {best_y2_model}")

if best_y1_model == 'Random Forest':
    y1_pred = rf_y1.predict(X_test)
else:
    y1_pred = linear_y1.predict(test_data[feature_cols])

if best_y2_model == 'Random Forest':
    y2_pred = rf_y2.predict(X_test)
else:
    y2_pred = linear_y2.predict(test_data[feature_cols])

submission = pd.DataFrame({
    'id': test_data['id'],
    'Y1': y1_pred,
    'Y2': y2_pred
})

submission.to_csv('preds.csv', index=False)
print("Predictions saved to: preds.csv")

######################## 9. VISUALIZATIONS ########################
print("\n\n9. VISUALIZATIONS\n")
plot_all(results, rf_y1_importance, rf_y2_importance, y1_linear_cv, y1_rf_cv, y2_linear_cv, y2_rf_cv)

print("\nANALYSIS COMPELTE")
print(f"Best combined performance: {results['Combined'].iloc[0]:.4f}")

# print("\nColumn names:")
# print(train_data.columns.tolist())

# print("\nInfo about Y1, Y2:")
# print(train_data[['Y1','Y2']].describe())

# y1_mean = train_data['Y1'].mean()
# y2_mean = train_data['Y2'].mean()

# print(f"\nBaseline predictions:")
# print(f"Y1 mean: {y1_mean}")
# print(f"Y2 mean: {y2_mean}")

# print("\nTop 3 correlations with Y1:")
# feature_cols = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
# correlations_y1 = train_data[feature_cols].corrwith(train_data['Y1']).sort_values(ascending=False)
# print(correlations_y1.head(14))

# print("\nTop 3 correlations with Y2:")
# correlations_y2 = train_data[feature_cols].corrwith(train_data['Y1']).sort_values(ascending=False)
# print(correlations_y2.head(14))

# print(f"\nTest data shape: {test_data.shape}")
# print("Test data columns:", test_data.columns.tolist())

