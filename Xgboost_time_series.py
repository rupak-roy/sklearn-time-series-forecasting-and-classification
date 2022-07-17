from sktime.forecasting.compose import make_reduction
import xgboost

# Create an exogenous dataframe indicating the month
X = pd.DataFrame({'month': y.index.month}, index=y.index)
X = pd.get_dummies(X.astype(str), drop_first=True)

X_train, X_test = temporal_train_test_split(X, test_size=36)

regressor = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)
forecaster = make_reduction(regressor, window_length=12, strategy="recursive")

# Fit and predict
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=fh, X=X_test)

# Evaluate
mean_absolute_percentage_error(y_test, y_pred)
#0.10052889328976747

# Plot predictions with training and test data
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], x_label='Date', y_label='ariline passenger');
