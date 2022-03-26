from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score

dataset = load_iris()

X = dataset.data
Y = dataset.target

# documentation link: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
clf = MLPClassifier(
    hidden_layer_sizes=(4,2),
    activation="relu", 
    solver="sgd", 
    batch_size=10, 
    learning_rate_init=0.01, 
    learning_rate="constant", 
    max_iter=100
)
clf.fit(X, Y)

weightArr = clf.coefs_
biasArr = clf.intercepts_

print("Jumlah layer: {}".format(clf.n_layers_))
print("Jumlah iterasi: {}".format(clf.n_iter_))
print("Jumlah fitur: {}".format(clf.n_features_in_))
print("Jumlah output: {}".format(clf.n_outputs_))

# Print array weight untuk setiap layer
print("Weight Input: \n", weightArr[0])
print("Bias Input: \n", biasArr[0])

print("Weight Hidden Layer: \n", weightArr[1])
print("Bias Hidden Layer: \n", biasArr[1])

print("Weight Output: \n", weightArr[2])
print("Bias Output: \n", biasArr[2])
# Y_pred = clf.predict(X)

# dataset_clf_accuracy = accuracy_score(Y, Y_pred)
# dataset_clf_f1 = f1_score(Y, Y_pred, average='weighted')

# print(f'Accuracy score = {dataset_clf_accuracy}')
# print(f'F1 Score = {dataset_clf_f1}')