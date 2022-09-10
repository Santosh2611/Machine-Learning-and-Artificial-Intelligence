import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
from numpy.linalg import eig # Compute the eigenvalues and right eigenvectors of a square array
import pandas as pd  # Library for working with data sets
import seaborn as sns # Provides high level API to visualize data

"""
There is a unicode character '\u0332', COMBINING LOW LINE*, which acts as an underline on the character that precedes it in a string. The center() method will center align the string, using a specified character (space is default) as the fill character.
"""

def print_csv_file(df, heading):
    print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    print(df)

def standardize_the_data_set():

    print("\nMean:")

    mean_X2 = df['X2'].mean()
    mean_X3 = df['X3'].mean()
    mean_X4 = df['X4'].mean()
    mean_Y = df['Y'].mean()

    print("X2. The age of house in years: " + str(mean_X2))
    print("X3. The distance to nearest MRT station in meters: " + str(mean_X3))
    print("X4. The number of convenience stores within walking distance: " + str(mean_X4))
    print("Y. House price per local unit area: " + str(mean_Y))

    print("\nStandard Deviation:")

    std_X2 = df['X2'].std()
    std_X3 = df['X3'].std()
    std_X4 = df['X4'].std()
    std_Y = df['Y'].std()

    print("X2. The age of house in years: " + str(std_X2))
    print("X3. The distance to nearest MRT station in meters: " + str(std_X3))
    print("X4. The number of convenience stores within walking distance: " + str(std_X4))
    print("Y. House price per local unit area: " + str(std_Y))

    heading = "Step 1 - Standardize the dataset (Z-Score Normalization)"
    for i in df.index:
        df.loc[i, 'X2'] = ((df.loc[i, 'X2']) - mean_X2)/std_X2
        df.loc[i, 'X3'] = ((df.loc[i, 'X3']) - mean_X3)/std_X3
        df.loc[i, 'X4'] = ((df.loc[i, 'X4']) - mean_X4)/std_X4
        df.loc[i, 'Y'] = ((df.loc[i, 'Y']) - mean_Y)/std_Y    
    print_csv_file(df, heading)
    
    standardized_dataset = df
    return standardized_dataset

# Calculate the covariance matrix for the whole dataset:-
def covariance_matrix():
    
    size = df.shape[1] # Get the shape of the DataFrame, which is a tuple where the first element is a number of rows and the second is the number of columns.
    covariance_matrix_1D = [0] * (size * size)

    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X2'] * df.loc[i, 'X2']
    covariance_matrix_1D[0] = sum / len(df)  # var(f1)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X2'] * df.loc[i, 'X3']
    covariance_matrix_1D[1] = sum / len(df)  # cov(f1,f2)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X2'] * df.loc[i, 'X4']
    covariance_matrix_1D[2] = sum / len(df)  # cov(f1,f3)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X2'] * df.loc[i, 'Y']
    covariance_matrix_1D[3] = sum / len(df)  # cov(f1,f4)

    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X3'] * df.loc[i, 'X2']
    covariance_matrix_1D[4] = sum / len(df)  # cov(f2,f1)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X3'] * df.loc[i, 'X3']
    covariance_matrix_1D[5] = sum / len(df)  # var(f2)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X3'] * df.loc[i, 'X4']
    covariance_matrix_1D[6] = sum / len(df)  # cov(f2,f3)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X3'] * df.loc[i, 'Y']
    covariance_matrix_1D[7] = sum / len(df)  # cov(f2,f4)

    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X4'] * df.loc[i, 'X2']
    covariance_matrix_1D[8] = sum / len(df)  # cov(f3,f1)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X4'] * df.loc[i, 'X3']
    covariance_matrix_1D[9] = sum / len(df)  # cov(f3,f2)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X4'] * df.loc[i, 'X4']
    covariance_matrix_1D[10] = sum / len(df)  # var(f3)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'X4'] * df.loc[i, 'Y']
    covariance_matrix_1D[11] = sum / len(df)  # cov(f3,f4)

    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'Y'] * df.loc[i, 'X2']
    covariance_matrix_1D[12] = sum / len(df)  # cov(f4,f1)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'Y'] * df.loc[i, 'X3']
    covariance_matrix_1D[13] = sum / len(df)  # cov(f4,f2)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'Y'] * df.loc[i, 'X4']
    covariance_matrix_1D[14] = sum / len(df)  # cov(f4,f3)
    sum = 0
    for i in df.index:
        sum = sum + df.loc[i, 'Y'] * df.loc[i, 'Y']
    covariance_matrix_1D[15] = sum / len(df)  # var(f4)

    covariance_matrix_multidim = np.reshape(covariance_matrix_1D, (size, size))
    print("\nThe covariance matrix is as follows:\n{}".format(
        covariance_matrix_multidim))
    return covariance_matrix_multidim

# Determine explained variance:-
def determine_explained_variance(eigenvalues):
    
    total_eigenvalues = sum(eigenvalues)
    explained_variance = [(i/total_eigenvalues) for i in sorted(eigenvalues, reverse=True)]
    cumulative_explained_variance = np.cumsum(explained_variance) # Return the cumulative sum of the elements along a given axis
    
    plt.bar(range(0,len(explained_variance)), explained_variance, alpha=0.5, align='center', label='Individual Explained Variance')
    plt.step(range(0,len(cumulative_explained_variance)), cumulative_explained_variance, where='mid',label='Cumulative Explained Variance')
    
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Convert CSV data into multidimensional array:-
def convert_csv_array():
    size = df.shape[1]
    single_dimensional_array = []
    for i in df.index:
        single_dimensional_array.append(df.loc[i, 'X2'])
        single_dimensional_array.append(df.loc[i, 'X3'])
        single_dimensional_array.append(df.loc[i, 'X4'])
        single_dimensional_array.append(df.loc[i, 'Y'])
    multi_dimensional_array = np.reshape(single_dimensional_array, (len(df), size))
    return multi_dimensional_array

# Driver Code: main() ; Execution starts here.

print("\nThere are three regressor variables and one response variable (namely, y):")
print("X2 - Age of House in year(s)")
print("X3 - Distance to Nearest MRT station in meter(s)")
print("X4 - Number of Convenience Stores within Walking Distance")
print("Y - House Price per Local Unit Area")

heading = "Original Data Set"
df = pd.read_csv("Real Estate Data Set.csv")
print_csv_file(df, heading)

# Step 1 - Standardize the dataset:-
standardized_dataset = standardize_the_data_set()
print("\nSince we have standardized the dataset, so the mean for each feature is 0 and the standard deviation is 1.\n")

heading = "Step 2 - Calculate the covariance matrix for the features in the dataset"
print('{:s}'.format('\u0332'.join(heading.center(100))))
covariance_matrix_multidim = covariance_matrix()

print("\nNow, we will check the co-relation between our scaled dataset using a heat map.  The correlation between various features is given by the corr() function and then the heat map is plotted by the heatmap() function. The colour scale on the side of the heatmap helps determine the magnitude of the co-relation.")

print("\nIn our example, we can clearly see that a darker shade represents less co-relation while a lighter shade represents more co-relation. The diagonal of the heatmap represents the co-relation of a feature with itself – which is always 1.0, thus, the diagonal of the heatmap is of the highest shade.")

sns.heatmap(standardized_dataset.corr())
plt.title("Co-relation between features before Principal Component Analysis (PCA)")
plt.grid(True)
plt.tight_layout()
plt.show()

heading = "Step 3 - Calculate and sort the eigenvalues and eigenvectors for the covariance matrix"
print("\n")
print('{:s}'.format('\u0332'.join(heading.center(100))))
eigenvalues, eigen_vectors = eig(covariance_matrix_multidim)

index = eigenvalues.argsort()[::-1] # Returns the indices that would sort an array
eigenvalues = eigenvalues[index]
eigen_vectors = eigen_vectors[:, index]

print("\nSorted Eigenvalues: ", eigenvalues)
print("\nCorresponding Eigen Vectors:\n{}".format(eigen_vectors))
determine_explained_variance(eigenvalues)

heading = "Step 4 - Pick k eigenvalues and form a matrix of eigenvectors"
print("\n")
print('{:s}'.format('\u0332'.join(heading.center(100))))

eigenvectors_1D = eigen_vectors.flatten() # Return a copy of the array (matrix) collapsed into one dimension.

while True:
    top = int(input("Enter the number of top eigen vectors to transform the original matrix: "))
    if top > df.shape[1]:
        print("Invalid Input! Please try again.")
    else:
        break

top_eigenvectors_1D = []  # NULL Single Dimensional Array Declaration
for i in range(int(len(eigenvectors_1D)/len(eigen_vectors)) * top):
    top_eigenvectors_1D.append(eigenvectors_1D[i])
top_eigenvectors_multidim = np.reshape(top_eigenvectors_1D, (len(eigen_vectors), top))
print("\nIf we choose the top " + str(top) + " eigenvectors, the matrix will look like this:\n{}".format(top_eigenvectors_multidim))

heading = "Step 5 - Tranform the Original Matrix"
print("\n")
print('{:s}'.format('\u0332'.join(heading.center(100))))

feature_matrix = convert_csv_array()
transformed_data = np.dot(feature_matrix, top_eigenvectors_multidim) # Matrix Multiplication

print("\nThe transformed data is as follows:\n")
for i in transformed_data:
    print(i)

print("\nNow that we have applied PCA and obtained the reduced feature set, we will check the co-relation between various Principal Components, again by using a heatmap.")

new_dataframe = pd.DataFrame(transformed_data) 
sns.heatmap(new_dataframe.corr())
plt.title("Co-relation between features after Principal Component Analysis (PCA)")
plt.grid(True)
plt.tight_layout()
plt.show()
