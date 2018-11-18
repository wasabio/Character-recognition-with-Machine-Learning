from sklearn import naive_bayes
from sklearn import tree
from sklearn.neural_network import MLPClassifier # Neural Network : MultiLayer Perceptron with back propagation
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

# Some experimentation with different hyper parameters
'''
# Decision Tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Decision Tree, criterion = Gini impurity, splitter = best)  : {}".format(accuracy))

classifier = tree.DecisionTreeClassifier()
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Decision Tree, criterion = Gini impurity, splitter = best)  : {}".format(accuracy))

classifier = tree.DecisionTreeClassifier(splitter="random")
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Decision Tree, criterion = entropy, splitter = random)  : {}".format(accuracy))

classifier = tree.DecisionTreeClassifier(splitter="random")
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Decision Tree, criterion = entropy, splitter = random)  : {}".format(accuracy))

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Decision Tree, criterion = entropy, splitter = best)  : {}".format(accuracy))

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Decision Tree, criterion = entropy, splitter = best)  : {}".format(accuracy))

# Naive Bayes
classifier = naive_bayes.BernoulliNB()
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Naive Bayes, Bernoulli, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.BernoulliNB()
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Naive Bayes, Bernoulli, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.GaussianNB()
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Naive Bayes, Gaussian, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.GaussianNB()
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Naive Bayes, Gaussian, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB()
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Naive Bayes, Multinomial, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB()
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Naive Bayes, Multinomial, smooth = 1)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB(alpha=20)
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Naive Bayes, Multinomial, smooth = 100)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB(alpha=20)
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Naive Bayes, Multinomial, smooth = 100)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB(alpha=1.0e-10)
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Naive Bayes, Multinomial, no smooth)  : {}".format(accuracy))

classifier = naive_bayes.MultinomialNB(alpha=1.0e-10)
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Naive Bayes, Multinomial, no smooth)  : {}".format(accuracy))

# MultiLayer Perceptron
classifier = MLPClassifier(hidden_layer_sizes=(5, 2))
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Neural network (MLP), Dimension = 5,2)  : {}".format(accuracy))

classifier = MLPClassifier(hidden_layer_sizes=(5, 2))
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Neural network (MLP), Dimension = 5,2)  : {}".format(accuracy))

classifier = MLPClassifier(hidden_layer_sizes=(50, 20))
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Neural network (MLP), Dimension = 50,20)  : {}".format(accuracy))

classifier = MLPClassifier(hidden_layer_sizes=(50, 20))
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Neural network (MLP), Dimension = 50,20)  : {}".format(accuracy))

classifier = MLPClassifier(hidden_layer_sizes=(100, 40))
classifier.fit(featuresTrainDS1, labelsTrainDS1)
accuracy = accuracy_score(labelsValDS1, classifier.predict(featuresValDS1))
print("Accuracy (DS1, Neural network (MLP), Dimension = 100,40)  : {}".format(accuracy))

classifier = MLPClassifier(hidden_layer_sizes=(100, 40))
classifier.fit(featuresTrainDS2, labelsTrainDS2)
accuracy = accuracy_score(labelsValDS2, classifier.predict(featuresValDS2))
print("Accuracy (DS2, Neural network (MLP), Dimension = 100,40)  : {}".format(accuracy))
'''
#Some experimentation with different hyper parameters - END

# Training classifier (option 1)
'''
classifierDT1 = tree.DecisionTreeClassifier(criterion='entropy')
classifierDT1.fit(featuresTrainDS1, labelsTrainDS1)

classifierNB1 = naive_bayes.MultinomialNB(alpha=0.5)
classifierNB1.fit(featuresTrainDS1, labelsTrainDS1)

classifierMLP1 = MLPClassifier(alpha=0.5, hidden_layer_sizes=(200, 80), random_state=1)
classifierMLP1.fit(featuresTrainDS1, labelsTrainDS1)

classifierDT2 = tree.DecisionTreeClassifier(criterion='entropy')
classifierDT2.fit(featuresTrainDS2, labelsTrainDS2)

classifierNB2 = naive_bayes.MultinomialNB(alpha=0.5)
classifierNB2.fit(featuresTrainDS2, labelsTrainDS2)

classifierMLP2 = MLPClassifier(alpha=0.5, hidden_layer_sizes=(200, 80), random_state=1)
classifierMLP2.fit(featuresTrainDS2, labelsTrainDS2)
'''
# Training classifier - End

# Loading an existing trained classifier (option 2)

with open(os.path.join(path, "savedModels", "ds1-dt.pkl"), 'rb') as file:
    classifierDT1 = pickle.load(file)
with open(os.path.join(path, "savedModels", "ds1-nb.pkl"), 'rb') as file:
    classifierNB1 = pickle.load(file)
with open(os.path.join(path, "savedModels", "ds1-mlp.pkl"), 'rb') as file:
    classifierMLP1 = pickle.load(file)
with open(os.path.join(path, "savedModels", "ds2-dt.pkl"), 'rb') as file:
    classifierDT2 = pickle.load(file)
with open(os.path.join(path, "savedModels", "ds2-nb.pkl"), 'rb') as file:
    classifierNB2 = pickle.load(file)
with open(os.path.join(path, "savedModels", "ds2-mlp.pkl"), 'rb') as file:
    classifierMLP2 = pickle.load(file)

# Loading an existing trained classifier - END


# Printing & saving

print(" - ACCURACY FOR CLASSIFIERS GENERATING OUTPUT FILES - ")
val_predicted = classifierNB1.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print("Accuracy for DS1 with Naive Bayes : {}".format(accuracy))
test_predicted = classifierNB1.predict(featuresTestDS1)
write_results_file("ds1Val-nb.csv", val_predicted)
write_results_file("ds1Test-nb.csv", test_predicted)
save_model(classifierNB1, "ds1-nb.pkl")

val_predicted = classifierDT1.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print("Accuracy for DS1 with Decision Tree : {}".format(accuracy))
test_predicted = classifierNB1.predict(featuresTestDS1)
write_results_file("ds1Val-dt.csv", val_predicted)
write_results_file("ds1Test-dt.csv", test_predicted)
save_model(classifierDT1, "ds1-dt.pkl")

val_predicted = classifierMLP1.predict(featuresValDS1)
accuracy = accuracy_score(labelsValDS1, val_predicted)
print("Accuracy for DS1 with Neural Network : {}".format(accuracy))
test_predicted = classifierMLP1.predict(featuresTestDS1)
write_results_file("ds1Val-mlp.csv", val_predicted)
write_results_file("ds1Test-mlp.csv", test_predicted)
save_model(classifierMLP1, "ds1-mlp.pkl")

val_predicted = classifierNB2.predict(featuresValDS2)
accuracy = accuracy_score(labelsValDS2, val_predicted)
print("Accuracy for DS2 with Naive Bayes : {}".format(accuracy))
test_predicted = classifierNB2.predict(featuresTestDS2)
write_results_file("ds2Val-nb.csv", val_predicted)
write_results_file("ds2Test-nb.csv", test_predicted)
save_model(classifierNB2, "ds2-nb.pkl")

val_predicted = classifierDT2.predict(featuresValDS2)
accuracy = accuracy_score(labelsValDS2, val_predicted)
print("Accuracy for DS2 with Decision Tree : {}".format(accuracy))
test_predicted = classifierDT2.predict(featuresTestDS2)
write_results_file("ds2Val-dt.csv", val_predicted)
write_results_file("ds2Test-dt.csv", test_predicted)
save_model(classifierDT2, "ds2-dt.pkl")

val_predicted = classifierMLP2.predict(featuresValDS2)
accuracy = accuracy_score(labelsValDS2, val_predicted)
print("Accuracy for DS2 with Neural Network : {}".format(accuracy))
test_predicted = classifierMLP2.predict(featuresTestDS2)
write_results_file("ds2Val-mlp.csv", val_predicted)
write_results_file("ds2Test-mlp.csv", test_predicted)
save_model(classifierMLP2, "ds2-mlp.pkl")

# Printing & saving - END
