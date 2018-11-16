from sklearn.externals import joblib
from sklearn import naive_bayes
from sklearn import tree
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
# DS1
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
# DS2
with open(os.path.join(path, "datasets", "ds2", "ds2Info.csv")) as file:
    data = [line.split(',') for line in file.read().split('\n')][1:]
    numbersDS2 = [d[:-1] for d in data]
    charactersDS2 = [d[-1] for d in data]

with open(os.path.join(path, "datasets", "ds2", "ds2Train.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresTrainDS2 = [[int(element) for element in row][:-1] for row in data]
    labelsTrainDS2 = [[int(element) for element in row][-1] for row in data]

with open(os.path.join(path, "datasets", "ds2", "ds2Val.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresValDS2 = [[int(element) for element in row][:-1] for row in data]
    labelsValDS2 = [[int(element) for element in row][-1] for row in data]

with open(os.path.join(path, "datasets", "ds2", "ds2Test.csv"), 'r') as file:
    data = [line.split(',') for line in file.read().split('\n')][:-1]
    featuresTestDS2 = [[int(element) for element in row] for row in data]
# Reading csv files for DS1 and DS2 - END

# Training classifier (option 1)
classifierDT1 = tree.DecisionTreeClassifier()
classifierDT1.fit(featuresTrainDS1, labelsTrainDS1)

classifierNB1 = naive_bayes.MultinomialNB(alpha=0.5)
classifierNB1.fit(featuresTrainDS1, labelsTrainDS1)

classifierDT2 = tree.DecisionTreeClassifier()
classifierDT2.fit(featuresTrainDS2, labelsTrainDS2)

classifierNB2 = naive_bayes.MultinomialNB(alpha=0.5)
classifierNB2.fit(featuresTrainDS2, labelsTrainDS2)
# Training classifier - End

# Loading an existing trained classifier (option 2)
'''
with open(os.path.join(path, "savedModels", "ds1-nb.pkl"), 'rb') as file:
    classifierNB1 = pickle.load(file)
'''
# Loading an existing trained classifier - END


# Printing & saving
val_predicted = classifierDT1.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print("Accuracy for DS1 with Decision Tree : {}".format(accuracy))
test_predicted = classifierNB1.predict(featuresTestDS1)
write_results_file("ds1Val-dt.csv", val_predicted)
write_results_file("ds1Test-dt.csv", test_predicted)
save_model(classifierDT1, "ds1-dt.pkl")

val_predicted = classifierNB1.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print("Accuracy for DS1 with Naive Bayes : {}".format(accuracy))
test_predicted = classifierNB1.predict(featuresTestDS1)
write_results_file("ds1Val-nb.csv", val_predicted)
write_results_file("ds1Test-nb.csv", test_predicted)
save_model(classifierNB1, "ds1-nb.pkl")

val_predicted = classifierDT2.predict(featuresValDS2)
accuracy = accuracy_score(labelsValDS2, val_predicted)
print("Accuracy for DS2 with Decision Tree : {}".format(accuracy))
test_predicted = classifierNB2.predict(featuresTestDS2)
write_results_file("ds2Val-dt.csv", val_predicted)
write_results_file("ds2Test-dt.csv", test_predicted)
save_model(classifierDT1, "ds2-dt.pkl")

val_predicted = classifierNB2.predict(featuresValDS2)
accuracy = accuracy_score(labelsValDS2, val_predicted)
print("Accuracy for DS2 with Naive Bayes : {}".format(accuracy))
test_predicted = classifierNB2.predict(featuresTestDS2)
write_results_file("ds2Val-nb.csv", val_predicted)
write_results_file("ds2Test-nb.csv", test_predicted)
save_model(classifierNB1, "ds2-nb.pkl")
# Printing & saving - END
