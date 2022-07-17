
#pip install sktime
#forecasting

from sktime.forecasting.all import *

y = load_airline()
y_train, y_test = temporal_train_test_split(y)

fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
 
y_pred = forecaster.predict(fh)