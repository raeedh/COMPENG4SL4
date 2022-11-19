import numpy as np
from statistics import mode
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def load_data(train_split, split_seed):
	"""
	train_split = percentage of set that should be split into training set
	split_seed = random generator seed used for splitting data to training and test sets
	
	load Wisconsin breast cancer data set and split into training and test sets
	"""
	X, y = load_breast_cancer(return_X_y=True)

	# Split data into training and test split
	# training-test split and seed are inputs from function
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, random_state=split_seed)

	# Perform feature standardization
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	return X_train, X_test, y_train, y_test

def batch_gradient_logreg(x, t, seed, alpha = 0.001, iterations = 100000):
	"""
	x = x data from set
	y = target data from set
	initial_w = initial conditions for predictor, w^(0)

	Returns coefficients of predictor in ndarray
	"""
	w = np.random.random(x.shape[1])

	for i in range(iterations):
		y = 1 / (1 + np.exp(-(x @ w)))
		w = w - (alpha/len(t))*(x.T @ (y - t))
	# print(w)
	return w

def kneighbour_kfold(X_train, y_train, kf_train, kf_test):
	"""
	Perform K-Fold Cross-Validation with k-neighbour classifier
	"""

	# Split training sets for K-fold from original training set
	T1_X_train, T1_y_train = X_train[kf_train[0]], y_train[kf_train[0]]
	T2_X_train, T2_y_train = X_train[kf_train[1]], y_train[kf_train[1]]
	T3_X_train, T3_y_train = X_train[kf_train[2]], y_train[kf_train[2]]
	T4_X_train, T4_y_train = X_train[kf_train[3]], y_train[kf_train[3]]
	T5_X_train, T5_y_train = X_train[kf_train[4]], y_train[kf_train[4]]
	# Split testing set for K-fold from original training set
	T1_X_test, T1_y_test = X_train[kf_test[0]], y_train[kf_test[0]]
	T2_X_test, T2_y_test = X_train[kf_test[1]], y_train[kf_test[1]]
	T3_X_test, T3_y_test = X_train[kf_test[2]], y_train[kf_test[2]]
	T4_X_test, T4_y_test = X_train[kf_test[3]], y_train[kf_test[3]]
	T5_X_test, T5_y_test = X_train[kf_test[4]], y_train[kf_test[4]]

	T1_dist, T2_dist, T3_dist, T4_dist, T5_dist = [], [], [], [], []
	for r in range(len(kf_test[0])):
		T1_dist.append(np.linalg.norm(T1_X_train - T1_X_test[r], axis=1))
		T2_dist.append(np.linalg.norm(T2_X_train - T2_X_test[r], axis=1))
		T3_dist.append(np.linalg.norm(T3_X_train - T3_X_test[r], axis=1))
		T4_dist.append(np.linalg.norm(T4_X_train - T4_X_test[r], axis=1))
		T5_dist.append(np.linalg.norm(T5_X_train - T5_X_test[r], axis=1))
	
	err = []
	for k in range(1,6):
		T1_pred, T2_pred, T3_pred, T4_pred, T5_pred = [], [], [], [], []
		for r in range(len(kf_test[0])):
			T1_calc = 0 if sum(T1_y_train[T1_dist[r].argsort()[:k]]) <= (k/2) else 1
			T1_pred.append(T1_calc)
			T2_calc = 0 if sum(T2_y_train[T2_dist[r].argsort()[:k]]) <= (k/2) else 1
			T2_pred.append(T2_calc)
			T3_calc = 0 if sum(T3_y_train[T3_dist[r].argsort()[:k]]) <= (k/2) else 1
			T3_pred.append(T3_calc)
			T4_calc = 0 if sum(T4_y_train[T4_dist[r].argsort()[:k]]) <= (k/2) else 1
			T4_pred.append(T4_calc)
			T5_calc = 0 if sum(T5_y_train[T5_dist[r].argsort()[:k]]) <= (k/2) else 1
			T5_pred.append(T5_calc)
		err_calc = [sum(np.bitwise_xor(T1_pred, T1_y_test)) / len(T1_y_test), sum(np.bitwise_xor(T2_pred, T2_y_test)) / len(T2_y_test), sum(np.bitwise_xor(T3_pred, T3_y_test)) / len(T3_y_test), sum(np.bitwise_xor(T4_pred, T4_y_test)) / len(T4_y_test), sum(np.bitwise_xor(T5_pred, T5_y_test)) / len(T5_y_test)]
		print("The cross-validation error (misclassification rate) for k =", k, "with 5-fold cross-validation is:", np.mean(err_calc))
		err.append(np.mean(err_calc))

	return err.index(min(err)) + 1

def kneighbour_pred(X_test, X_train, y_train, k):
	"""
	Predict k-neighbour neighbours
	"""

	dist, pred = [], []
	for r in range(len(X_test)):
		dist.append(np.linalg.norm(X_train - X_test[r], axis=1))
	for r in range(len(X_test)):
		calc = 0 if sum(y_train[dist[r].argsort()[:k]]) <= (k/2) else 1
		pred.append(calc)

	return pred

def kneighbour_kfold_scikit(X_train, y_train, kf_train, kf_test):
	"""
	Perform K-Fold Cross-Validation with k-neighbour classifier
	"""

	# Split training sets for K-fold from original training set
	T1_X_train, T1_y_train = X_train[kf_train[0]], y_train[kf_train[0]]
	T2_X_train, T2_y_train = X_train[kf_train[1]], y_train[kf_train[1]]
	T3_X_train, T3_y_train = X_train[kf_train[2]], y_train[kf_train[2]]
	T4_X_train, T4_y_train = X_train[kf_train[3]], y_train[kf_train[3]]
	T5_X_train, T5_y_train = X_train[kf_train[4]], y_train[kf_train[4]]

	total = []

	for k in range(1,6):
		n1 = KNeighborsClassifier(n_neighbors=k).fit(T1_X_train, T1_y_train)
		n2 = KNeighborsClassifier(n_neighbors=k).fit(T2_X_train, T2_y_train)
		n3 = KNeighborsClassifier(n_neighbors=k).fit(T3_X_train, T3_y_train)
		n4 = KNeighborsClassifier(n_neighbors=k).fit(T4_X_train, T4_y_train)
		n5 = KNeighborsClassifier(n_neighbors=k).fit(T5_X_train, T5_y_train)

		n1_targets = n1.predict(X_train[kf_test[0]])
		n2_targets = n2.predict(X_train[kf_test[1]])
		n3_targets = n3.predict(X_train[kf_test[2]])
		n4_targets = n4.predict(X_train[kf_test[3]])
		n5_targets = n5.predict(X_train[kf_test[4]])

		
		total_calc = [sum(np.bitwise_xor(n1_targets, y_train[kf_test[0]])), sum(np.bitwise_xor(n2_targets, y_train[kf_test[1]])), sum(np.bitwise_xor(n3_targets, y_train[kf_test[2]])), sum(np.bitwise_xor(n4_targets, y_train[kf_test[3]])), sum(np.bitwise_xor(n5_targets, y_train[kf_test[4]]))] 
		print("The cross-validation error (misclassification rate) for k =", k, "with 5-fold cross-validation is:", np.mean(total_calc) / len(kf_test[0]))
		total.append(np.mean(total_calc))

	return total.index(min(total)) + 1

def main():
	# Load Wisconsin breast cancer data, perform feature standarization  and split into 80-20 training-test sets
	# Use 8200 (last 4 digits of student ID for random seed)
	X_train, X_test, y_train, y_test = load_data(0.8, 8200)
	print("The Wisconsin breast cancer data set and split to 80-20 training-test sets using a seed of 8200")
	print("")

	# Get indeces for K-fold Cross-validation for later use
	kf = KFold(n_splits = 5)
	kf_train = []
	kf_test = []

	for train, test in kf.split(X_train):
		kf_train.append(train)
		kf_test.append(test)
	
	# Create threshold list for logistic regression
	threshold = list(np.arange(0.05,1,0.05))

	# Implement logistic regression
	w = batch_gradient_logreg(X_train, y_train, 8200)
	print("\nLogistic Regression is implemented")
	y_prob = 1 / (1 + np.exp(-(X_test @ w)))
	y_prob = 1 - np.array(y_prob)
	for i in threshold:
		y_targ = 1 * (y_prob < 1 - i)
		y_targ_inv = np.where(y_targ | 1, y_targ^1, y_targ)
		y_test_inv = np.where(y_test | 1, y_test^1, y_test)
		logreg_p = sum(y_targ_inv * y_test_inv) / len(np.where(y_targ_inv == 0)[0])
		logreg_r = sum(y_targ_inv * y_test_inv)  / len(np.where(y_test == 0)[0])
		logreg_f1 = (2 * logreg_p * logreg_r) / (logreg_p + logreg_r)
		print("The precision, recall, and F1 score when the threshold is", i, "were calculated to be:")
		print("precision: ", logreg_p)
		print("recall: ", logreg_r)
		print("f1 score: ", logreg_f1)
		logreg_misclass = np.bitwise_xor(y_targ, y_test)
		print("The test error (misclassification rate) is:", sum(logreg_misclass) / len(y_test))
		# print(sklearn_logreg_prob)
		print("")

	# Implementation of logistic regression provided in scikit-learn
	# Create LogisticRegression object and fit to training data
	print("\nLogistic Regression is implemented using the implementation provided in scikit-learn")
	clf = LogisticRegression().fit(X_train, y_train)
	# Predict using logistic regression model on test set
	sklearn_logreg_targets = clf.predict(X_test)
	sklearn_logreg_prob = clf.predict_proba(X_test)
	for i in threshold:
		sklearn_logreg_targets = 1 * (sklearn_logreg_prob[:, 0] < 1-i)
		sklearn_logreg_targets_inv = np.where(sklearn_logreg_targets | 1, sklearn_logreg_targets^1, sklearn_logreg_targets)
		y_test_inv = np.where(y_test | 1, y_test^1, y_test)
		sklearn_logreg_p = sum(sklearn_logreg_targets_inv * y_test_inv) / len(np.where(sklearn_logreg_targets == 0)[0])
		sklearn_logreg_r = sum(sklearn_logreg_targets_inv * y_test_inv)  / len(np.where(y_test == 0)[0])
		sklearn_logreg_f1 = (2 * sklearn_logreg_p * sklearn_logreg_r) / (sklearn_logreg_p + sklearn_logreg_r)
		print("The precision, recall, and F1 score when the threshold is", i, "were calculated to be:")
		print("precision: ", sklearn_logreg_p)
		print("recall: ", sklearn_logreg_r)
		print("f1 score: ", sklearn_logreg_f1)
		sklearn_logreg_misclass = np.bitwise_xor(sklearn_logreg_targets, y_test)
		print("The test error (misclassification rate) is:", sum(sklearn_logreg_misclass) / len(y_test))
		# print(sklearn_logreg_prob)
		print("")

	# Own implementation of k-nearest neighbour classifier calculation to get best k
	print("Implement of k-nearest neighbour is implemented after finding best k for k = 1-5 using 5-fold cross-validation")
	k_neigh = kneighbour_kfold(X_train, y_train, kf_train, kf_test)
	print("The best k was found to be", k_neigh)
	# Get test predictors for with k-nearest neighbour classifier for best k
	kneigh_pred = kneighbour_pred(X_test, X_train, y_train, k_neigh)
	kneigh_pred = np.array(kneigh_pred)
	# Calculate misclassification rate
	kneigh_misclass = np.bitwise_xor(kneigh_pred, y_test)
	print("The test error (misclassification rate) with k =", k_neigh, "is: ", sum(kneigh_misclass) / len(y_test))
	# Calculate F1 score
	kneigh_pred_inv = np.where(kneigh_pred | 1, kneigh_pred^1, kneigh_pred)
	y_test_inv = np.where(y_test | 1, y_test^1, y_test)
	kneigh_p = sum(kneigh_pred_inv * y_test_inv) / len(np.where(kneigh_pred_inv == 0)[0])
	kneigh_r = sum(kneigh_pred_inv * y_test_inv)  / len(np.where(y_test == 0)[0])
	kneigh_f1 = (2 * kneigh_p * kneigh_r) / (kneigh_p + kneigh_r)
	print("The F1 score is:", kneigh_f1)
	print("")

	# Implementation of k-nearest neighbour classifier provided in scikit-learn
	print("k-nearest neighbour is implemented using the implementation provided in scikit-learn")
	scikit_learn_k = kneighbour_kfold_scikit(X_train, y_train, kf_train, kf_test)
	print("The best k was found to be", scikit_learn_k)
	neigh = KNeighborsClassifier(n_neighbors=scikit_learn_k).fit(X_train, y_train)
	# Predict using KNeighborsClassifier on test set
	sklearn_kneigh_targets = neigh.predict(X_test)
	# Calculate misclassification rate
	sklearn_kneigh_misclass = np.bitwise_xor(sklearn_kneigh_targets, y_test)
	print("The test error (misclassification rate) with k =", scikit_learn_k, "is: ", sum(sklearn_kneigh_misclass) / len(y_test))
	# Calculate F1 score
	sklearn_kneigh_targets_inv = np.where(sklearn_kneigh_targets | 1, sklearn_kneigh_targets^1, sklearn_kneigh_targets)
	y_test_inv = np.where(y_test | 1, y_test^1, y_test)
	kneigh_scikit_p = sum(sklearn_kneigh_targets_inv * y_test_inv) / len(np.where(sklearn_kneigh_targets_inv == 0)[0])
	kneigh_scikit_r = sum(sklearn_kneigh_targets_inv * y_test_inv)  / len(np.where(y_test == 0)[0])
	kneigh_scikit_f1 = (2 * kneigh_scikit_p * kneigh_scikit_r) / (kneigh_scikit_p + kneigh_scikit_r)
	print("The F1 score is:", kneigh_scikit_f1)
	print("")
	
	

if __name__ == '__main__':
	main()