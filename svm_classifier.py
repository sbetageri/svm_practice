import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv('Skin_NonSkin.txt')

label = dataframe['S']

features = dataframe[['B', 'G', 'R']]

features_train, features_test, labels_train, labels_test = train_test_split(features, label, random_state=9)

classifier = svm.SVC()

classifier.fit(features_train, labels_train)

pred = classifier.predict(features_test)

print(accuracy_score(labels_test, pred))
