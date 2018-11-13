from sklearn.externals import joblib
from sklearn import naive_bayes


with open('datasets/ds1/ds1Info.csv', 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][1:]
    numbers = [d[:-1] for d in data]
    characters = [d[-1] for d in data]

with open('datasets/ds1/ds1Train.csv', 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresDS1 = [[int(element) for element in row][:-1] for row in data]
    labelsDS1 = [[int(element) for element in row][-1] for row in data]

classifier = naive_bayes.MultinomialNB()
classifier.fit(featuresDS1, labelsDS1)
validation_predicted = classifier.predict(validation_features)
