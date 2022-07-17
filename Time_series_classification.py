from sktime.classification.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_arrow_head(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
#0.8867924528301887

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cm

report = classification_report(y_test,y_pred,output_dict=False)