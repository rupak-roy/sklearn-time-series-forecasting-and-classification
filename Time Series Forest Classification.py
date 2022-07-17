tsf = TimeSeriesForestClassifier(
     estimator=time_series_tree, 
     n_estimators=100,
     criterion="entropy",
     bootstrap=True,
     oob_score=True,
     random_state=1,
     n_jobs=-1,
 )
 
tsf = TimeSeriesForestClassifier()
tsf.fit(X_train, y_train)

tsf.score(X_test, y_test)

fi = tsf.feature_importances_
fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
fi.plot(ax=ax)
ax.set(xlabel="Time", ylabel="Feature importance");