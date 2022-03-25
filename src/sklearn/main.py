from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# documentation link: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
clf = MLPClassifier(
    activation="relu", batch_size=10, learning_rate_init=0.01, max_iter=100
)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

dataset_clf_accuracy = accuracy_score(Y_test, Y_pred)
dataset_clf_f1 = f1_score(Y_test, Y_pred, average='weighted')

print(f'Accuracy score = {dataset_clf_accuracy}')
print(f'F1 Score = {dataset_clf_f1}')
