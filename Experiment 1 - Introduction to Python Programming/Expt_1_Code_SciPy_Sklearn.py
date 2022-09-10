from scipy import constants
print("\nConstants in SciPy:")
print(constants.pi) 

print("\nRoots of an Equation:")
from scipy.optimize import root
from math import cos
def eqn(x):
  return x + cos(x)
myroot = root(eqn, 0)
print(myroot.x) # Find root of the equation x + cos(x)
print("\n")
print(myroot) # Print all information about the solution (not just x which is the root)

print("\nDataset Loading:")
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
print("Feature names:", feature_names)
print("Target names:", target_names)
print("\nFirst 10 rows of X:\n", X[:10])

print("\nSplitting the Dataset:")
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
