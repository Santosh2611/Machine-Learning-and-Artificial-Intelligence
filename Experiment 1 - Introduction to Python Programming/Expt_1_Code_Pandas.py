import pandas as pd # Library for working with data sets

print("\nPandas Series:")
a = [1, 7, 2]
myvar = pd.Series(a)
print(myvar)
print(myvar[0]) # Return the first value of the series

print("\nCreate Labels:")
a = [1, 7, 2]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar)
print(myvar["y"]) # Return the value of "y"

print("\nKey/Value Objects as Series:")
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)
print(myvar)

print("\nDataFrames:")
data = {
            "calories": [420, 380, 390],
            "duration": [50, 40, 45]
        }
myvar = pd.DataFrame(data)
print(myvar)

print("\nRead CSV Files:")
df = pd.read_csv('data.csv')
print(df.to_string()) 

print("\nViewing the Data: ")
df = pd.read_csv('data.csv')
print(df.head(10)) 

print("\nInfo About the Data:")
df = pd.read_csv('data.csv')
print(df.info()) 
