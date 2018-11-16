from sklearn.externals import joblib
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import os
import pickle


# FUNCTIONS
def write_results_file(filename, values):
    with open(os.path.join(path, "results", filename), 'w') as f:
        for x in range(len(values)):
            f.write('%d,%d\n' % (x + 1, values[x]))


def save_model(model_classifier, model_name):
    with open(os.path.join(path, "savedModels", model_name), 'wb') as f:
        pickle.dump(model_classifier, f)
# FUNCTIONS - END


path = os.path.dirname(os.getcwd())

# Reading csv files for DS1 and DS2
with open(os.path.join(path, "datasets", "ds1", "ds1Info.csv")) as file:
    data = [line.split(',') for line in file.read().split('\n')][1:]
    numbersDS1 = [d[:-1] for d in data]
    charactersDS1 = [d[-1] for d in data]

with open(os.path.join(path, "datasets", "ds1", "ds1Train.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresTrainDS1 = [[int(element) for element in row][:-1] for row in data]
    labelsTrainDS1 = [[int(element) for element in row][-1] for row in data]

with open(os.path.join(path, "datasets", "ds1", "ds1Val.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresValDS1 = [[int(element) for element in row][:-1] for row in data]
    labelsValDS1 = [[int(element) for element in row][-1] for row in data]

with open(os.path.join(path, "datasets", "ds1", "ds1Test.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresTestDS1 = [[int(element) for element in row] for row in data]
# Reading csv files for DS1 and DS2 - END

# Training classifier (option 1)
'''
classifier = naive_bayes.MultinomialNB(alpha=0.5)
classifier.fit(featuresTrainDS1, labelsTrainDS1)
'''
# Training classifier - End

# Loading an existing trained classifier (option 2)
with open(os.path.join(path, "savedModels", "ds1-nb.pkl"), 'rb') as file:
    classifier = pickle.load(file)
# Loading an existing trained classifier - END

val_predicted = classifier.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print(accuracy)
test_predicted = classifier.predict(featuresTestDS1)
write_results_file("ds1Val-nb.csv", val_predicted)
write_results_file("ds1Test-nb.csv", test_predicted)
save_model(classifier, "ds1-nb.pkl")
