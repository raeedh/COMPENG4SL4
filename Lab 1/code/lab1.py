import numpy as np
import math
from sklearn.preprocessing import StandardScaler

# Generate the training and validation sets
def generate_data(training_size, valiation_size, random_seed):
	X_train = np.linspace(0.,1.,training_size) # training set
	X_valid = np.linspace(0.,1.,valiation_size) # validation set
	np.random.seed(random_seed) # Seed is last 4 digits of McMaster student ID
	t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(training_size) 
	t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(valiation_size)

	# Return training and validation sets
	return X_train, X_valid, t_train, t_valid

# Calculate least squares regression
# Returns coefficients of predictor in ndarray
def least_square_regression(x, t, M, regularization_lambda=0):
	X = np.tile(x.reshape(-1,1),M+1)
	B = np.diag(regularization_lambda*np.ones(M+1))
	for D in range(M+1):
		X[:,D] = X[:,D]**D
	return np.matmul(np.linalg.inv(np.add(np.matmul(X.T, X),B)),np.matmul(X.T,t.T))

# Calculate least squares regression with regularization
def least_square_regression_regularization(x, t, M, regularization_lambda=0):
	X = np.tile(x.reshape(-1,1),M+1)
	B = np.diag(regularization_lambda*np.ones(M+1))
	for D in range(M+1):
		X[:,D] = X[:,D]**D
	return np.matmul(np.linalg.inv(np.add(np.matmul(X.T, X),B)),np.matmul(X.T,t.T))

# Generate predictor coefficients for M from 0 to M (input)
def least_square_regression_M(x, t, M):
	w = []
	for i in range(M+1):
		w.append(least_square_regression(x, t, i))
	return w

# Calculate training/validation error
# x/t = training/validationg set, N = number of elements, predictor = coefficients of predictor in numpy.ndarray
def calc_error(x, t, N, predictor):
	cum_error = 0
	for i in range(N):
		cum = 0
		for n, d in enumerate(predictor):
			cum += d*x[i]**n
		cum_error += (t[i] - cum)**2
	return cum_error / N

# who knows
def idk(X_train, X_valid, M):
	XX_train = np.tile(X_train.reshape(-1,1),M)
	XX_valid = np.tile(X_valid.reshape(-1,1),M)
	for D in range(M):
		XX_train[:,D] = XX_train[:,D]**(D+1)
	sc = StandardScaler()
	XX_train = sc.fit_transform(XX_train)
	XX_valid = sc.transform(XX_valid)

	return XX_train, XX_valid

def main():
	# Generate the training and validation sets
	X_train, X_valid, t_train, t_valid = generate_data(10, 100, 8200)

	# Calculator predictors for all M for 0-9 (inclusive)
	w = least_square_regression_M(X_train, t_train, 9)

	# Generate predictors for M = 9 with regularization
	lambdas = [math.exp(-10), math.exp(-15), math.exp(-16), math.exp(-17), math.exp(-18), math.exp(-19), math.exp(-20), math.exp(-21), math.exp(-22), math.exp(-23), math.exp(-24), math.exp(-25), math.exp(-30), math.exp(-40), math.exp(-50)]
	w9_reg = []
	for val in lambdas:
		w9_reg.append(least_square_regression(X_train, t_train, 9, val))

	# Calculate training error for all predictors for M = 0-9
	training_error = []
	for i in range(len(w)):
		training_error.append(calc_error(X_train, t_train, 10, w[i]))
	# print(training_error)
	
	# Calculate validation error for all predictors for M = 0-9
	validation_error = []
	for i in range(len(w)):
		validation_error.append(calc_error(X_valid, t_valid, 10, w[i]))

	# Calculate training error for all predictors for M = 9 with regularization
	training_error_reg = []
	for i in range(len(w9_reg)):
		training_error_reg.append(calc_error(X_train, t_train, 10, w9_reg[i]))
	# print(training_error_reg)
	
	# Calculate validation error for all predictors for M = 9 with regularization
	validation_error_reg = []
	for i in range(len(w9_reg)):
		validation_error_reg.append(calc_error(X_valid, t_valid, 10, w9_reg[i]))
	# print(validation_error_reg)

	XX_train, XX_valid = idk(X_train, X_valid, 9)
	print(XX_train)

if __name__ == '__main__':
	main()