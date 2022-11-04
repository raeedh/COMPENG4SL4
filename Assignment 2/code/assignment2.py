import warnings
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold


def load_data(train_split, split_seed):
	"""
	train_split = percentage of set that should be split into training set
	split_seed = random generator seed used for splitting data to training and test sets
	
	load Boston housing data set and split into training and test sets
	"""

	# load Boston housing data set
	with warnings.catch_warnings():
		# You should probably not use this dataset.
		warnings.filterwarnings("ignore")
		X, y = load_boston(return_X_y=True)

	# Split data into training and test split
	# training-test split and seed are inputs from function
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, random_state=split_seed)
	return X_train, X_test, y_train, y_test

def least_square_regression(x, y, M):
	"""
	x = x data from set
	y = target data from set
	M = predictor capacity

	Calculate least squares regression
	Returns coefficients of predictor in ndarray
	Reuse from Assignment 1
	"""
	X = np.c_[np.ones(len(x)), x]
	return np.matmul(np.linalg.inv(np.matmul(X.T, X)),np.matmul(X.T,y.T))

def calc_error(x, y, w):
	"""
	x = x data from set
	y = target data from set
	w = predictor coefficients
	
	Calculate and return mean squared error
	"""

	X = np.c_[np.ones(len(x)), x]

	y_pred = (X * w).sum(axis=1)
	return np.mean((y - y_pred)**2)

def kfold_calc(X, y, kf_train, kf_test, S, f_total=13):
	"""
	Calculate K-fold Cross-validation error
	"""

	cross_valid_err = []
	f_list = []
	for f in range(f_total):
		if ((f+1) in S):
			continue
		
		S_test = [x - 1 for x in S]  + [f]
		
		# Split training sets for K-fold from original training set
		T1_X_train, T1_y_train = X[np.ix_(kf_train[0],S_test)], y[kf_train[0]]
		T2_X_train, T2_y_train = X[np.ix_(kf_train[1],S_test)], y[kf_train[1]]
		T3_X_train, T3_y_train = X[np.ix_(kf_train[2],S_test)], y[kf_train[2]]
		T4_X_train, T4_y_train = X[np.ix_(kf_train[3],S_test)], y[kf_train[3]]
		T5_X_train, T5_y_train = X[np.ix_(kf_train[4],S_test)], y[kf_train[4]]

		# Split testing set for K-fold from original training set
		T1_X_test, T1_y_test = X[np.ix_(kf_test[0],S_test)], y[kf_test[0]]
		T2_X_test, T2_y_test = X[np.ix_(kf_test[1],S_test)], y[kf_test[1]]
		T3_X_test, T3_y_test = X[np.ix_(kf_test[2],S_test)], y[kf_test[2]]
		T4_X_test, T4_y_test = X[np.ix_(kf_test[3],S_test)], y[kf_test[3]]
		T5_X_test, T5_y_test = X[np.ix_(kf_test[4],S_test)], y[kf_test[4]]

		w1 = least_square_regression(T1_X_train, T1_y_train, len(S))
		w2 = least_square_regression(T2_X_train, T2_y_train, len(S))
		w3 = least_square_regression(T3_X_train, T3_y_train, len(S))
		w4 = least_square_regression(T4_X_train, T4_y_train, len(S))
		w5 = least_square_regression(T5_X_train, T5_y_train, len(S))


		b = [calc_error(T1_X_test,T1_y_test,w1),calc_error(T2_X_test,T2_y_test,w2),calc_error(T3_X_test,T3_y_test,w3),calc_error(T4_X_test,T4_y_test,w4),calc_error(T5_X_test,T5_y_test,w5)]
		cross_valid_err.append(np.mean(b))
		f_list.append(f+1)
		
	print("The cross-validation errors for k =", len(S_test), "are", cross_valid_err)
	print("The best feature is f" + str(f_list[cross_valid_err.index(min(cross_valid_err))]) + " with cross-validation error of " + str(min(cross_valid_err)))
	S.append(f_list[cross_valid_err.index(min(cross_valid_err))])

def kcalc_basis(X, y, kf_train, kf_test, S, basis):
	"""
	Calculate K-fold for a basis.
	basis = 1 for f(x) =  sqrt(x)
	"""

	S_test = [x - 1 for x in S]
	
	# Split training sets for K-fold from original training set
	T1_X_train, T1_y_train = X[np.ix_(kf_train[0],S_test)], y[kf_train[0]]
	T2_X_train, T2_y_train = X[np.ix_(kf_train[1],S_test)], y[kf_train[1]]
	T3_X_train, T3_y_train = X[np.ix_(kf_train[2],S_test)], y[kf_train[2]]
	T4_X_train, T4_y_train = X[np.ix_(kf_train[3],S_test)], y[kf_train[3]]
	T5_X_train, T5_y_train = X[np.ix_(kf_train[4],S_test)], y[kf_train[4]]
	# Split testing set for K-fold from original training set
	T1_X_test, T1_y_test = X[np.ix_(kf_test[0],S_test)], y[kf_test[0]]
	T2_X_test, T2_y_test = X[np.ix_(kf_test[1],S_test)], y[kf_test[1]]
	T3_X_test, T3_y_test = X[np.ix_(kf_test[2],S_test)], y[kf_test[2]]
	T4_X_test, T4_y_test = X[np.ix_(kf_test[3],S_test)], y[kf_test[3]]
	T5_X_test, T5_y_test = X[np.ix_(kf_test[4],S_test)], y[kf_test[4]]

	if (basis == 1):
		T1_X_train = np.sqrt(T1_X_train)
		T2_X_train = np.sqrt(T2_X_train)
		T3_X_train = np.sqrt(T3_X_train)
		T4_X_train = np.sqrt(T4_X_train)
		T5_X_train = np.sqrt(T5_X_train)

		T1_X_test = np.sqrt(T1_X_test)
		T2_X_test = np.sqrt(T2_X_test)
		T3_X_test = np.sqrt(T3_X_test)
		T4_X_test = np.sqrt(T4_X_test)
		T5_X_test = np.sqrt(T5_X_test)

		w1 = least_square_regression(T1_X_train, T1_y_train, len(S))
		w2 = least_square_regression(T2_X_train, T2_y_train, len(S))
		w3 = least_square_regression(T3_X_train, T3_y_train, len(S))
		w4 = least_square_regression(T4_X_train, T4_y_train, len(S))
		w5 = least_square_regression(T5_X_train, T5_y_train, len(S))

		b = [calc_error(T1_X_test,T1_y_test,w1),calc_error(T2_X_test,T2_y_test,w2),calc_error(T3_X_test,T3_y_test,w3),calc_error(T4_X_test,T4_y_test,w4),calc_error(T5_X_test,T5_y_test,w5)]
		return np.mean(b)
	elif (basis == 2):
		T1_X_train = np.log(T1_X_train + 1)
		T2_X_train = np.log(T2_X_train + 1)
		T3_X_train = np.log(T3_X_train + 1)
		T4_X_train = np.log(T4_X_train + 1)
		T5_X_train = np.log(T5_X_train + 1)

		T1_X_test = np.log(T1_X_test + 1)
		T2_X_test = np.log(T2_X_test + 1)
		T3_X_test = np.log(T3_X_test + 1)
		T4_X_test = np.log(T4_X_test + 1)
		T5_X_test = np.log(T5_X_test + 1)

		w1 = least_square_regression(T1_X_train, T1_y_train, len(S))
		w2 = least_square_regression(T2_X_train, T2_y_train, len(S))
		w3 = least_square_regression(T3_X_train, T3_y_train, len(S))
		w4 = least_square_regression(T4_X_train, T4_y_train, len(S))
		w5 = least_square_regression(T5_X_train, T5_y_train, len(S))

		b = [calc_error(T1_X_test,T1_y_test,w1),calc_error(T2_X_test,T2_y_test,w2),calc_error(T3_X_test,T3_y_test,w3),calc_error(T4_X_test,T4_y_test,w4),calc_error(T5_X_test,T5_y_test,w5)]
		return np.mean(b)


def model_err_calc(X_train, y_train, X_test, y_test, S):
	"""
	Calculate the error for a model
	"""
	S_test = [x - 1 for x in S]
	X = X_train[:,S_test]

	w = least_square_regression(X, y_train, len(S))
	
	print("The test error for k = " + str(len(S)) + " is:", calc_error(X_test[:,S_test], y_test, w))

def basis_model_err_calc(X_train, y_train, X_test, y_test, S, basis):
	"""
	Calculate the error for a model with basis expansion
	"""
	S_test = [x - 1 for x in S]
	X = X_train[:,S_test]

	if (basis == 1):
		X = np.sqrt(X)
		XX_test = np.sqrt(X_test)
	elif (basis == 2):
		X = np.log(X + 1)
		XX_test = np.log(X_test + 1)

	w = least_square_regression(X, y_train, len(S))
	
	print("The test error for k = " + str(len(S)) + " with basis expansion with basis function", basis, "is:", calc_error(XX_test[:,S_test], y_test, w))

def main():
	# Load Boston housing data and split into 80-20 training-test sets
	# Use 8200 (last 4 digits of student ID for random seed)
	X_train, X_test, y_train, y_test = load_data(0.8, 8200)

	# Get indeces for K-fold Cross-validation for later use
	kf = KFold(n_splits = 5)
	kf_train = []
	kf_test = []

	for train, test in kf.split(X_train):
		kf_train.append(train)
		kf_test.append(test)

	# Create empty list for S
	S = []

	# Perform K-fold Cross-Validation and Basis Expansion for k = 1-13
	for f in range(13):
		kfold_calc(X_train, y_train, kf_train, kf_test, S)
		model_err_calc(X_train, y_train, X_test, y_test, S)
		b1 = kcalc_basis(X_train, y_train, kf_train, kf_test, S, 1)
		b2 = kcalc_basis(X_train, y_train, kf_train, kf_test, S, 2)
		print("For basis function 1, f(x) = sqrt(x), the cross-validation error is:", b1)
		print("For basis function 2, f(x) = log(x + 1), the cross-validation error is:", b2)
		print("We select basis function", 1 if b1 < b2 else 2, "as it produces the lower cross-validation error.")
		basis_model_err_calc(X_train, y_train, X_test, y_test, S, 1 if b1 < b2 else 2)
		print("")

	print("The final subset S for k = 13 is:", S)
	

if __name__ == '__main__':
	main()