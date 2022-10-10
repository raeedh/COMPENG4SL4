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

# Calculate average error between targets and true function f_true(x)
def avg_true_error(x, t):
	cum = 0
	for n, val in enumerate(x):
		cum += abs(math.sin(4*math.pi*val) - t[n])
	return cum / len(x)

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
def calc_error(x, t, predictor):
	cum_error = 0
	for i in range(len(x)):
		cum = 0
		for n, d in enumerate(predictor):
			cum += d*x[i]**n
		cum_error += (t[i] - cum)**2
	return cum_error / len(x)

def calc_error_regularization(x, t, lmbda, predictor):
	cum_error = 0
	for i in range(len(x)):
		cum = 0
		for n, d in enumerate(predictor):
			cum += d*x[i]**n
		cum_error += (t[i] - cum)**2
	regularized_cost = cum_error / len(x)
	for val in predictor:
		regularized_cost += lmbda*(val**2)
	return regularized_cost 

# Feature standardization
def feature_standardization(X_train, X_valid, M):
	XX_train = np.tile(X_train.reshape(-1,1),M)
	XX_valid = np.tile(X_valid.reshape(-1,1),M)
	for D in range(M):
		XX_train[:,D] = XX_train[:,D]**(D+1)
	sc = StandardScaler()
	XX_train = sc.fit_transform(XX_train)
	XX_valid = sc.transform(XX_valid)

	return XX_train, XX_valid

# Export data for plotting
def export_data(x, t, filename):
	f = open(filename, "w")
	f.write("$x$\t$t$\n")

	for n, val in enumerate(x):
		f.write(str(val))
		f.write("\t")
		f.write(str(t[n]))
		f.write("\n")
	
	f.close()

def main():
	# Generate the training and validation sets
	# First argument is number of points in training set
	# Second argument is number of points in validation set
	# Third argument is seed for random number generator (use last 4 digits of student ID)
	print("Generated training and validation sets of size 10 and 100 with random number generator seed 8200.\n")
	X_train, X_valid, t_train, t_valid = generate_data(10, 100, 8200)

	# Calculate average error between validation targets and true function f_true(x)
	print("Calculated average error between validation targets and true function ftrue(x):")
	print(avg_true_error(X_valid, t_valid))
	print("\n")

	# Export training and validation sets for plotting
	export_data(X_train, t_train, "training_set.dat")
	export_data(X_valid, t_valid, "validation_set.dat")

	# Calculator predictors for all M for 0-9 (inclusive)
	w = least_square_regression_M(X_train, t_train, 9)

	# For each predictor, print coefficients
	print("The predictors were determined and the coefficients for each M (in ascending powers) are:")
	for i in range(10):
		print("M = " + str(i) + ", " + str(w[i]))
	print("\n")

	# Calculate training error for all predictors for M = 0-9
	training_error = []
	for i in range(len(w)):
		training_error.append(calc_error(X_train, t_train, w[i]))
	# Print results
	print("The training error for all predictors was calculated:")
	for i in range(10):
		print("M = " + str(i) + ", " + str(training_error[i]))
	print("The lowest training error occured when M = " + str(training_error.index(min(training_error))) + ".")
	print("\n")

	# Calculate validation error for all predictors for M = 0-9
	validation_error = []
	for i in range(len(w)):
		validation_error.append(calc_error(X_valid, t_valid, w[i]))
	# Print results
	print("The validation error for all predictors was calculated:")
	for i in range(10):
		print("M = " + str(i) + ", " + str(validation_error[i]))
	print("The lowest validation error occured when M = " + str(validation_error.index(min(validation_error))) + ".")
	print("\n")

	# Generate predictors for M = 9 with regularization for various lambda
	lambdas = [math.exp(0), math.exp(-1), math.exp(-2), math.exp(-3), math.exp(-4), math.exp(-5), math.exp(-6), math.exp(-7), math.exp(-8), math.exp(-9), math.exp(-10), math.exp(-11), math.exp(-12), math.exp(-13), math.exp(-14), math.exp(-15), math.exp(-16), math.exp(-17), math.exp(-18), math.exp(-19), math.exp(-20), math.exp(-21), math.exp(-22), math.exp(-23), math.exp(-24), math.exp(-25), math.exp(-30), math.exp(-40), math.exp(-50)]
	w9_reg = []
	for val in lambdas: # use additional argument λ for least_square_regression to enable regularization (default value is 0 if no argument passed in)
		w9_reg.append(least_square_regression(X_train, t_train, 9, val))
	print("Linear regression with regularization was done for various \u03BB for M = 9.")
	print("\n")

	# Calculate training error for all predictors for M = 9 with regularization
	training_error_reg = []
	for i in range(len(w9_reg)):
		training_error_reg.append(calc_error_regularization(X_train, t_train, lambdas[i], w9_reg[i]))
	# Print results
	print("The training error for various \u03BB was calculated:")
	for i in range(len(training_error_reg)):
		print("ln \u03BB = " + str(math.log(lambdas[i])) + ", " + str(training_error_reg[i]))
	print("The lowest training error with regularization is " + str(min(training_error_reg)) + " when ln \u03BB = " + str(math.log(lambdas[training_error_reg.index(min(training_error_reg))])) + ".")
	print("\n")
	
	# Calculate validation error for all predictors for M = 9 with regularization
	validation_error_reg = []
	for i in range(len(w9_reg)):
		validation_error_reg.append(calc_error_regularization(X_valid, t_valid, lambdas[i], w9_reg[i]))
	# Print results
	print("The validation error for various \u03BB was calculated:")
	for i in range(len(validation_error_reg)):
		print("ln \u03BB = " + str(math.log(lambdas[i])) + ", " + str(validation_error_reg[i]))
	print("The lowest validation error with regularization is " + str(min(validation_error_reg)) + " when ln \u03BB = " + str(math.log(lambdas[validation_error_reg.index(min(validation_error_reg))])) + ".")
	print("The predictor coefficients for this linear regression is:")
	print(w9_reg[validation_error_reg.index(min(validation_error_reg))])
	print("\n")

	print("We observe we reduce overfitting when \u03BB is between -10 and -25, an example is when ln \u03BB = -25 (the lowest validation error), for which the predictor coefficients are:")
	print(w9_reg[lambdas.index(math.exp(-25))])

	print("We observe the plot is underfitted when \u03BB is too high, an example is when ln \u03BB = 0, for which the predictor coefficients are:")
	print(w9_reg[lambdas.index(math.exp(0))])

	XX_train, XX_valid = feature_standardization(X_train, X_valid, 9)

	print(".")

if __name__ == '__main__':
	main()