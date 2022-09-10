# Importing Necessary Libraries:-
import matplotlib.pyplot as plt # Provides an implicit way of plotting
from matplotlib.colors import ListedColormap # Colormap object generated from a list of colors
import numpy as np # Support for large, multi-dimensional arrays and matrices
import pandas as pd  # Library for working with data sets
from sklearn.metrics import accuracy_score # Accuracy classification score
from sklearn.metrics import classification_report # Build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix # Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes (GaussianNB)
from sklearn.neighbors import KNeighborsClassifier # Classifier implementing the k-nearest neighbors vote
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
from sklearn.svm import SVC # C-Support Vector Classification

import warnings
warnings.filterwarnings('ignore') # Never print matching warnings

def print_csv_file(df, heading):
    print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    print(df)

# Function to split the dataset:-
def splitdataset(dataFrame):
    
    # Create feature and target arrays:-
    X = dataFrame.iloc[:, [1, 2, 6]].values # Data - Assume Independent Variable
    y = dataFrame.iloc[:, 3].values # Target - Assume Dependent Variable
    
    # Split into training and test set:-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test

def k_nearest_neighbor(dataFrame):
    
    X, y, X_train, X_test, y_train, y_test = splitdataset(dataFrame) # Function to split the dataset
    neighbors = np.arange(1, 11) # Return evenly spaced values within a given interval
    
    # Return a new array of given shape, without initializing entries:-
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
     
    # Loop over K values:-
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k) # Classifier implementing the k-nearest neighbors vote
        knn.fit(X_train, y_train) # Fit the k-nearest neighbors classifier from the training dataset
         
        # Compute training and test data accuracy - Return the mean accuracy on the given test data and labels:-
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
     
    # Generate plot:-     
    plt.xlabel('n_neighbors'); plt.ylabel('Accuracy'); plt.title("K-Nearest Neighbor (KNN) - Model Accuracy")
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    plt.legend(); plt.grid(True); plt.show()
    
# Function to make predictions:-
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex:-
    y_pred = clf_object.predict(X_test)
    print("Predicted Values -"); print(y_pred)
    return y_pred
      
# Function to calculate accuracy:-
def cal_accuracy(y_test, y_pred):
      
    print("\nConfusion Matrix -\n", confusion_matrix(y_test, y_pred)) # Compute confusion matrix to evaluate the accuracy of a classification
      
    print ("\nAccuracy - ", accuracy_score(y_test,y_pred)*100) # Accuracy classification score
      
    print("Report -\n", classification_report(y_test, y_pred)) # Build a text report showing the main classification metrics
    
def visualize_result(classifier, x_set, y_set, classification_name, plot_name):
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01), np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01)) # Return coordinate matrices from coordinate vectors
    
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green'))) # Plot filled contours 
    plt.xlim(x1.min(), x1.max()) # Get or set the x limits of the current axes 
    plt.ylim(x2.min(), x2.max()) # Get or set the y limits of the current axes 
    
    for i, j in enumerate(np.unique(y_set)):  
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j) # A scatter plot of y vs. x with varying marker size and/or color  
    
    plt.xlabel("Age of House in Year(s)")  
    plt.ylabel("House Price per Local Unit Area")
    plt.title(str(classification_name) + " - " + str(plot_name)) 
    plt.legend(); plt.grid(True); plt.show()

def model_evaluation_selection(dataFrame):
    
    # Create feature and target arrays:-
    X = dataFrame.iloc[:, [1, 6]].values # Data - Assume Independent Variable
    y = dataFrame.iloc[:, 3].values # Target - Assume Dependent Variable
    
    # Split into training and test set:-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    # Feature Scaling:-  
    st_x = StandardScaler() # Standardize features by removing the mean and scaling to unit variance    
    X_train = st_x.fit_transform(X_train) # Fit to data, then transform it
    X_test = st_x.transform(X_test) # Perform standardization by centering and scaling
    
    return X_train, X_test, y_train, y_test

def naive_bayes_classifier(dataFrame):
    
    X_train, X_test, y_train, y_test = model_evaluation_selection(dataFrame)
    
    # Fitting Naive Bayes to the training set:-
    classifier = GaussianNB() # Gaussian Naive Bayes (GaussianNB)  
    classifier.fit(X_train, y_train) # Fit Gaussian Naive Bayes according to X, y    
    y_pred= classifier.predict(X_test) # Predict the test set result
    cal_accuracy(y_test, y_pred) # Function to calculate accuracy
    
    visualize_result(classifier, X_train, y_train, "Naive Bayes Classifier", "Training Set")
    visualize_result(classifier, X_test, y_test, "Naive Bayes Classifier", "Testing Set")
    
def support_vector_machine(dataFrame):
    
    X_train, X_test, y_train, y_test = model_evaluation_selection(dataFrame)
    
    # Fitting the SVM classifier to the training set:-
    classifier = SVC(kernel='linear', random_state=0) # C-Support Vector Classification  
    classifier.fit(X_train, y_train) # Fit the SVM model according to the given training data      
    y_pred = classifier.predict(X_test) # Predict the test set result
    cal_accuracy(y_test, y_pred) # Function to calculate accuracy
    
    visualize_result(classifier, X_train, y_train, "SVM Classifier", "Training Set")
    visualize_result(classifier, X_test, y_test, "SVM Classifier", "Testing Set")
    
def decision_tree(dataFrame):

    # Function to perform training with gini index:-
    def train_using_gini(X_train, X_test, y_train):
      
        # Create the classifier object:-
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5) # A decision tree classifier
      
        # Perform training:-
        clf_gini.fit(X_train, y_train) # Build a decision tree classifier from the training set (X, y)
        return clf_gini
    
    # Function to perform training with entropy:-
    def train_using_entropy(X_train, X_test, y_train):
      
        # Decision tree with entropy:-
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,    max_depth = 3, min_samples_leaf = 5) # A decision tree classifier
      
        # Perform training:-
        clf_entropy.fit(X_train, y_train) # Build a decision tree classifier from the training set (X, y)
        return clf_entropy
        
    X, y, X_train, X_test, y_train, y_test = splitdataset(dataFrame) # Function to split the dataset
    clf_gini = train_using_gini(X_train, X_test, y_train) # Function to perform training with gini index
    clf_entropy = train_using_entropy(X_train, X_test, y_train) # Function to perform training with entropy

    print("\n\t\t\t\t\t\t\t(i) Results Using Gini Index:-\n")      
    y_pred_gini = prediction(X_test, clf_gini) # Function to make predictions
    cal_accuracy(y_test, y_pred_gini) # Function to calculate accuracy
      
    print("\n\t\t\t\t\t\t\t(ii) Results Using Entropy:-\n")
    y_pred_entropy = prediction(X_test, clf_entropy) # Function to make predictions
    cal_accuracy(y_test, y_pred_entropy) # Function to calculate accuracy
    
# Driver Code: main():-
def main():
    print("\n")
    print("No - Serial Number")
    print("X2 - Age of House in Year(s)")
    print("X3 - Distance to Nearest MRT Station in Meter(s)")
    print("X4 - Number of Convenience Stores Within Walking Distance")
    print("X5 - Latitude Coordinates")
    print("X5 - Longitude Coordinates")
    print("Y - House Price per Local Unit Area")
    
    # Importing the data set:-
    heading = "Original Data Set"
    df = pd.read_csv("Real Estate Data Set.csv")
    print_csv_file(df, heading)
    
    print("\n"); heading = "K-Nearest Neighbor (KNN)"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    k_nearest_neighbor(df)
    
    print("\n"); heading = "Naive Bayes Classifier"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    naive_bayes_classifier(df)
    
    print("\n"); heading = "Decision Tree"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    decision_tree(df)
    
    print("\n"); heading = "Support Vector Machine (SVM)"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    support_vector_machine(df)

# Call main function ; Execution starts here.
if __name__=="__main__":
    main()
