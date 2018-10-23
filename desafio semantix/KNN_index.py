from sklearn import neighbors
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
import numpy as np

def KNN_index(data_previous):

    le1 = preprocessing.LabelEncoder()
    le1.fit(data_previous.iloc[:,0])
    data_previous.iloc[:,0] = le1.transform(data_previous.iloc[:,0])

    le2 = preprocessing.LabelEncoder()
    le2.fit(data_previous.iloc[:,1])
    data_previous.iloc[:,1] = le2.transform(data_previous.iloc[:,1])

    data = data_previous.iloc[:,0]
    target = data_previous.iloc[:,1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    np.reshape(data.values,(-1,1)),target.values, test_size=0.8, random_state=42)

    my_classifier = neighbors.KNeighborsClassifier()
    my_classifier.fit(X_train, y_train)
    predictions = my_classifier.predict(X_test)

    # Accuracy tree
    from sklearn.metrics import accuracy_score
    print("Acurácia da previsão, usando a campanha anterior:",accuracy_score(y_test, predictions))
