import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

def load_data(test_split, split_seed):
	"""
	train_split = percentage of set that should be split into training set
	split_seed = random generator seed used for splitting data to training and test sets
	
	load Wisconsin breast cancer data set and split into training and test sets
	"""
	dataset = pd.read_csv('Data/spambase.data', header=None)
	X = dataset.iloc[:, :-1].values
	t = dataset.iloc[:, -1].values

	X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=test_split, random_state=split_seed)

	return X_train, X_test, y_train, y_test

def decision_tree_kfold_scikit(X_train, y_train, kf_train, kf_test):
	"""
	Perform K-Fold Cross-Validation with Decision Tree Classifier
	"""

	# Split training sets for K-fold from original training set
	T1_X_train, T1_y_train = X_train[kf_train[0]], y_train[kf_train[0]]
	T2_X_train, T2_y_train = X_train[kf_train[1]], y_train[kf_train[1]]
	T3_X_train, T3_y_train = X_train[kf_train[2]], y_train[kf_train[2]]
	T4_X_train, T4_y_train = X_train[kf_train[3]], y_train[kf_train[3]]
	T5_X_train, T5_y_train = X_train[kf_train[4]], y_train[kf_train[4]]

	total = []

	for k in range(2,401):
		n1 = DecisionTreeClassifier(max_leaf_nodes = k, random_state = 8200).fit(T1_X_train, T1_y_train)
		n2 = DecisionTreeClassifier(max_leaf_nodes = k, random_state = 8200).fit(T2_X_train, T2_y_train)
		n3 = DecisionTreeClassifier(max_leaf_nodes = k, random_state = 8200).fit(T3_X_train, T3_y_train)
		n4 = DecisionTreeClassifier(max_leaf_nodes = k, random_state = 8200).fit(T4_X_train, T4_y_train)
		n5 = DecisionTreeClassifier(max_leaf_nodes = k, random_state = 8200).fit(T5_X_train, T5_y_train)

		n1_targets = n1.predict(X_train[kf_test[0]])
		n2_targets = n2.predict(X_train[kf_test[1]])
		n3_targets = n3.predict(X_train[kf_test[2]])
		n4_targets = n4.predict(X_train[kf_test[3]])
		n5_targets = n5.predict(X_train[kf_test[4]])
		
		total_calc = [sum(np.bitwise_xor(n1_targets, y_train[kf_test[0]])), sum(np.bitwise_xor(n2_targets, y_train[kf_test[1]])), sum(np.bitwise_xor(n3_targets, y_train[kf_test[2]])), sum(np.bitwise_xor(n4_targets, y_train[kf_test[3]])), sum(np.bitwise_xor(n5_targets, y_train[kf_test[4]]))] 
		print("The cross-validation error (misclassification rate) for max leaf nodes =", k, "with 5-fold cross-validation is:", np.mean(total_calc) / len(kf_test[0]))
		total.append(np.mean(total_calc) / len(kf_test[0]))

	return total.index(min(total)) + 2, min(total)

def bagging_classifier_scikit(X_train, y_train, X_test, y_test):
	"""
	Generate 50 Bagging Classifiers with 50 to 2500 predictors (in increments of 50) in the ensemble
	"""

	for n in range(50, 2501, 50):
		clf = BaggingClassifier(n_estimators = n, random_state = 8200).fit(X_train, y_train)
		bagging_pred = clf.predict(X_test)
		err = sum(np.bitwise_xor(bagging_pred, y_test)) / len(y_test)
		print("The test error (misclassification rate) for n =", n, "predictors in the ensemble is:", err)

def random_forest_classifier_scikit(X_train, y_train, X_test, y_test):
	"""
	Generate 50 Random Forest Classifiers with 50 to 2500 predictors (in increments of 50) in the ensemble
	"""

	for n in range(50, 2501, 50):
		clf = RandomForestClassifier(n_estimators = n, random_state = 8200).fit(X_train, y_train)
		bagging_pred = clf.predict(X_test)
		err = sum(np.bitwise_xor(bagging_pred, y_test)) / len(y_test)
		print("The test error (misclassification rate) for n =", n, "predictors in the ensemble is:", err)

def adaboost_classifier_stump_scikit(X_train, y_train, X_test, y_test):
	"""
	Generate 50 Adaboost Classifiers with 50 to 2500 predictors (in increments of 50) in the ensemble and decision stumps in the base classifiers
	"""

	for n in range(50, 2501, 50):
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = n, random_state = 8200).fit(X_train, y_train)
		bagging_pred = clf.predict(X_test)
		err = sum(np.bitwise_xor(bagging_pred, y_test)) / len(y_test)
		print("The test error (misclassification rate) for n =", n, "predictors in the ensemble is:", err)

def adaboost_classifier_10_leaves_scikit(X_train, y_train, X_test, y_test):
	"""
	Generate 50 Adaboost Classifiers with 50 to 2500 predictors (in increments of 50) in the ensemble and at most 10 leaves in the base classifiers
	"""

	for n in range(50, 2501, 50):
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=10), n_estimators = n, random_state = 8200).fit(X_train, y_train)
		bagging_pred = clf.predict(X_test)
		err = sum(np.bitwise_xor(bagging_pred, y_test)) / len(y_test)
		print("The test error (misclassification rate) for n =", n, "predictors in the ensemble is:", err)

def adaboost_classifier_no_restrict_scikit(X_train, y_train, X_test, y_test):
	"""
	Generate 50 Adaboost Classifiers with 50 to 2500 predictors (in increments of 50) in the ensemble and no restrictions for depth or node number in the base classifiers
	"""

	for n in range(50, 2501, 50):
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None, max_leaf_nodes=None), n_estimators = n, random_state = 8200).fit(X_train, y_train)
		bagging_pred = clf.predict(X_test)
		err = sum(np.bitwise_xor(bagging_pred, y_test)) / len(y_test)
		print("The test error (misclassification rate) for n =", n, "predictors in the ensemble is:", err)

def main():
	# Load Spambase data, perform feature standarization  and split into 80-20 training-test sets
	# Use 8200 (last 4 digits of student ID for random seed)
	X_train, X_test, y_train, y_test = load_data(0.33, 8200)
	print("The Spambase data set was loaded and split to 66-33 training-test sets using a seed of 8200")
	print("")

	# Get indeces for K-fold Cross-validation for later use
	kf = KFold(n_splits = 5)
	kf_train = []
	kf_test = []

	for train, test in kf.split(X_train):
		kf_train.append(train)
		kf_test.append(test)

	print("Decision tree clasiffiers with maximum number of leaves between 2 and 400 are trained and tested with 5-fold cross-validation.")
	dec_tree_max, dec_tree_max_cross_err = decision_tree_kfold_scikit(X_train, y_train, kf_train, kf_test)
	print("The lowest cross-validation error was achieved when the maximum number of leaves was", dec_tree_max, "leaves with a cross-validation error of", dec_tree_max_cross_err)
	print("")

	print("A model is trained on the training data set with", dec_tree_max, "maximum number of leaves.")
	dec_tree = DecisionTreeClassifier(max_leaf_nodes = dec_tree_max, random_state = 8200).fit(X_train, y_train)
	dec_tree_pred = dec_tree.predict(X_test)
	dec_tree_test_err = sum(np.bitwise_xor(dec_tree_pred, y_test)) / len(y_test)
	print("The test error (misclassification rate) with", dec_tree_max, "maximum number of leaves is", dec_tree_test_err)
	print("")

	print("50 bagging classifiers are trained and tested.")
	bagging_classifier_scikit(X_train, y_train, X_test, y_test)
	print("")

	print("50 random forest classifiers are trained and tested.")
	random_forest_classifier_scikit(X_train, y_train, X_test, y_test)
	print("")

	print("50 Adaboost classifiers with decision stumps are trained and tested.")
	adaboost_classifier_stump_scikit(X_train, y_train, X_test, y_test)
	print("")

	print("50 Adaboost classifiers with a maximum of 10 leaves are trained and tested.")
	adaboost_classifier_10_leaves_scikit(X_train, y_train, X_test, y_test)
	print("")

	print("50 Adaboost classifiers with no restrictions on depth or node number are trained and tested.")
	adaboost_classifier_no_restrict_scikit(X_train, y_train, X_test, y_test)
	print("")

if __name__ == '__main__':
	main()