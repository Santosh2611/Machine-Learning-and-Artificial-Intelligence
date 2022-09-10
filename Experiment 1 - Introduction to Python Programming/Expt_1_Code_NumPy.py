import numpy as np # Support for large, multi-dimensional arrays and matrices

print("\n")
# Create a NumPy ndarray Object:
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

print("\n")
# Create a 0-D array with value 42:
arr = np.array(42)
print(arr)

print("\n")
# Create a 1-D array containing the values 1,2,3,4,5:
arr = np.array([1, 2, 3, 4, 5])
print(arr)

print("\n")
# Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6:
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

print("\n")
# Access Array Elements:
arr = np.array([1, 2, 3, 4])
print(arr[0]) # Get the first element from the array
print(arr[1]) # Get the second element from the array
print(arr[2] + arr[3]) # Get third and fourth elements from the array and add them

print("\n")
# Access 2-D Arrays:
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print("2nd element on 1st row: ", arr[0, 1]) # Access the element on the 1st row, 2nd column
print("5th element on 2nd row: ", arr[1, 4]) # Access the element on the 2nd row, 5th column

print("\n")
# Access 3-D Arrays:
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2]) 

print("\n")
# Slicing arrays:
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5]) # Slice elements from index 1 to index 5 from the array
print(arr[4:]) # Slice elements from index 4 to the end of the array
print(arr[:4]) # Slice elements from the beginning to index 4 (not included)

print("\n")
# Negative Slicing:
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1]) 

print("\n")
# STEP:
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5:2]) # Return every other element from index 1 to index 5
print(arr[::2]) # Return every other element from the entire array
