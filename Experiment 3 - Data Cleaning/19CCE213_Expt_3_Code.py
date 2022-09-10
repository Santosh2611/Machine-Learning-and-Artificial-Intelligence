import numpy as np # Support for large, multi-dimensional arrays and matrices
import pandas as pd # Library for working with data sets

"""
There is a unicode character '\u0332', COMBINING LOW LINE*, which acts as an underline on the character that precedes it in a string. The center() method will center align the string, using a specified character (space is default) as the fill character.
"""

def print_csv_file(df): 
    print("\n")
    heading = "Read CSV File"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    print(df.to_string()) 

def clean_empty_cells(): # Clean Empty Cells
     
    def remove_rows(): # Remove Rows
        
        print("\nRemove Rows - This is usually OK, since data sets can be very big, and removing a few rows will not have a big impact on the result.")
        
        change = str(input("Do you want to change the original DataFrame? (y/n): "))
            
        if (change == 'N' or change == 'n'): # Return a new Data Frame with no empty cells
            new_df = df.dropna() # By default, the dropna() method returns a new DataFrame, and will not change the original.
            print_csv_file(new_df) 
                
        elif (change == 'Y' or change == 'y'): # Remove all rows with NULL values
            df.dropna(inplace = True) # Use the inplace = True argument
            print_csv_file(df)
            # Now, the dropna(inplace = True) will NOT return a new DataFrame, but it will remove all rows containg NULL values from the original DataFrame.
            
        else:
            print("Error: Invalid Input! Please try again.")
                      
    def replace_empty_values(): # Replace Empty Values
        
        print("\nReplace Empty Values - This way you do not have to delete entire rows just because of some empty cells. The fillna() method allows us to replace empty cells with a value.")
        
        value = input("Enter the value with which you want to replace the empty cell: ")
        df.fillna(value, inplace = True) # Replace NULL values with the number "value"
        print_csv_file(df)
                
    def replace_only_specified_columns(): # Replace Only For Specified Columns
    
        print("\nReplace Only For Specified Columns - To only replace empty values for one column, specify the column name for the DataFrame.")
        
        value = input("Enter the value with which you want to replace the empty cell in the column 'X2. The age of house in years': ")
        
        df['X2 house age'].fillna(value, inplace = True) 
        print_csv_file(df)

    def replace_using_mean(): # Replace Using Mean
        print("\nMean - The average value (the sum of all values divided by number of values).")
        mean = df['X2 house age'].mean()
        df['X2 house age'].fillna(mean, inplace = True) 
        print_csv_file(df)
            
    def replace_using_median(): # Replace Using Median
        print("\nMedian - The value in the middle, after you have sorted all values ascending.")
        median = df['X2 house age'].median()
        df['X2 house age'].fillna(median, inplace = True)
        print_csv_file(df)
    
    while True:  # This simulates a Do Loop
    
        print("\n")
        heading = "CLEAN EMPTY CELLS - MENU"
        print('{:s}'.format('\u0332'.join(heading.center(100))))
        
        choice = input(
            "   1. Remove Rows\n   2. Replace Empty Values\n   3. Replace Only For Specified Columns\n   4. Replace Using Mean\n   5. Replace Using Median\n   6. Exit\nEnter the number corresponding to the menu to implement the choice: ") # Menu Driven Implementation
        
        # str() returns the string version of the variable "choice"        
        if choice == str(1):
            remove_rows() # Remove Rows          
        elif choice == str(2):
            replace_empty_values() # Replace Empty Values           
        elif choice == str(3):
            replace_only_specified_columns() # Replace Only For Specified Columns
        elif choice == str(4):
            replace_using_mean() # Replace Using Mean
        elif choice == str(5):
            replace_using_median() # Replace Using Median
        elif choice == str(6):
            break  # Exit loop
        else:
            print("Error: Invalid Input! Please try again.")

def clean_data_wrong_format(): # Clean Data of Wrong Format
    
    print("\nIn our Data Frame, we have two cells with the wrong format. Check out row 8 and 14, the 'X1 transaction date' column should be a string that represents a date.")
    
    def convert_into_correct_format(): # Convert Into a Correct Format
        
        df['X1 transaction date'] = pd.to_datetime(df['X1 transaction date'])
        print_csv_file(df)
        
    def remove_rows(): # Remove Rows - The result from the converting in the example above gave us a NaT value, which can be handled as a NULL value, and we can remove the row by using the dropna() method.
        
        df.dropna(subset=['X1 transaction date'], inplace = True)
        print_csv_file(df)
    
    while True:  # This simulates a Do Loop
    
        print("\n")
        heading = "CLEAN DATA OF WRONG FORMAT - MENU"
        print('{:s}'.format('\u0332'.join(heading.center(100))))
    
        choice = input(
            "   1. Convert Into a Correct Format\n   2. Remove Rows\n   3. Exit\nEnter the number corresponding to the menu to implement the choice: ") # Menu Driven Implementation
        
        # str() returns the string version of the variable "choice".      
        if choice == str(1):
            
            print("\nLet's try to convert all cells in the 'X1. Transaction Date' column into dates.")
            convert_into_correct_format() # Convert Into a Correct Format          
        
        elif choice == str(2):
            
            print("\nAs you can see from the result, the date in row 14 was fixed, but the empty date in row 8 got a NaT (Not a Time) value, in other words, an empty value. One way to deal with empty values is simply removing the entire row.")
            remove_rows() # Remove Rows
        
        elif choice == str(3): 
            break  # Exit loop
        
        else:
            print("Error: Invalid Input! Please try again.")
        
def fix_wrong_data(): # Fix Wrong Data
    
    def convert_into_correct_format(): # Convert Into a Correct Format 
        df.loc[4, 'X4 number of convenience stores'] = 10
        print_csv_file(df)
        
    # Compute the qth quantile of the given data (array elements) along the specified axis.
    def print_five_number_summary_IQR_outlier(minimum, Q1, median, Q3, maximum):
        
        print("Minimum = ", minimum)
        print("Q1 quantile = ", Q1)
        print("Median =", median)
        print("Q3 quantile = ", Q3)
        print("Maximum =", maximum)
        
        IQR = Q3 - Q1
        print("Inter-Quartile Range (IQR) = ", IQR)
        outlier = 1.5 * IQR
        print("Outlier (1.5 X IQR) = ", outlier)
        df.loc[4, 'X4 number of convenience stores'] = outlier
        
    def calc_five_number_summary_variance_standard_deviation():
        
        min_X4 = df['X4 number of convenience stores'].min()
        Q1_X4 = np.quantile(df['X4 number of convenience stores'], .25)
        median_X4 = df['X4 number of convenience stores'].median()
        Q3_X4 = np.quantile(df['X4 number of convenience stores'], .75)
        max_X4 = df['X4 number of convenience stores'].max()
        print_five_number_summary_IQR_outlier(min_X4, Q1_X4, median_X4, Q3_X4, max_X4)
        
        var_X4 = df['X4 number of convenience stores'].var()
        print("Variance = ", var_X4)
        std_X4 = df['X4 number of convenience stores'].std()
        print("Standard Deviation = ", std_X4)
    
    def remove_rows(): # Remove Rows
        
        max_value = input("\nEnter the value above which the row should be deleted: ")
        for i in df.index: # Delete rows where "X4 number of convenience stores" is higher than "max_value"
            if df.loc[i, 'X4 number of convenience stores'] > int(max_value):
                df.drop(i, inplace = True)
        print_csv_file(df)
        
    while True:  # This simulates a Do Loop
        
        print("\n")
        heading = "FIX WRONG - MENU"
        print('{:s}'.format('\u0332'.join(heading.center(100))))
    
        choice = input(
            "   1. Replace Values\n   2. Remove Rows\n   3. Exit\nEnter the number corresponding to the menu to implement the choice: ") # Menu Driven Implementation
        
        # str() returns the string version of the variable "choice"        
        if choice == str(1):
            
            print("\nIn our example, it is most likely a typo, and the value should be '10' instead of '100', and we could just insert '10' in row 5.")
            convert_into_correct_format() # Convert Into a Correct Format
            
            print("\nFor small data sets, you might be able to replace the wrong data one by one, but not for big data sets. To replace wrong data for larger data sets you can create some rules, e.g. outliers.")            
            calc_five_number_summary_variance_standard_deviation()            
            print_csv_file(df)
            
        elif choice == str(2):
            
            print("\nRemove Rows - This way you do not have to find out what to replace them with, and there is a good chance you do not need them to do your analyses.")
            remove_rows() # Remove Rows
        
        elif choice == str(3): 
            break  # Exit loop
        else:
            print("Error: Invalid Input! Please try again.")

def remove_duplicates(): # Remove Duplicates

    print("\nTo discover duplicates, we can use the duplicated() method. The duplicated() method returns a Boolean value for each row:")
    print(df.duplicated()) # Returns True for every row that is a duplicate, othwerwise False.
    
    print("\nTo remove duplicates, use the drop_duplicates() method.")
    df.drop_duplicates(inplace = True) # Remove all duplicates
    # The (inplace = True) will make sure that the method does NOT return a new DataFrame, but it will remove all duplicates from the original DataFrame.  
    print_csv_file(df)
    
# Driver Code: main() ; Execution starts here. 

df = pd.read_csv("Real Estate Data Set.csv")
print_csv_file(df)

print("\n")
heading = "Identification of Response Variable & Regressor Variables"
print('{:s}'.format('\u0332'.join(heading.center(100))))

print("\nThere are three regressor variables (X1, X2, X4), namely:")
print("X1 - Transaction Date")
print("X2 - Age of House in Year(s)")
print("X4 - Number of Convenience Stores within Walking Distance")
    
print("\n")
heading = "Our Data Set"
print('{:s}'.format('\u0332'.join(heading.center(100))))

print("1. The data set contains some empty cells ('X1 transaction date' in row 9, and 'X2 house age' in row 7 and 10).")
print("2. The data set contains wrong format ('X1 transaction date' in row 15).")
print("3. The data set contains wrong data ('X4 number of convenience stores' in row 5).")
print("4. The data set contains duplicates (row 3 and 4).")

while True:  # This simulates a Do Loop
    
    print("\n")
    heading = "MAIN MENU"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    
    choice = input(
        "   1. Clean Empty Cells\n   2. Clean Data of Wrong Format\n   3. Fix Wrong Data\n   4. Remove Duplicates\n   5. Exit\nEnter the number corresponding to the menu to implement the choice: ") # Menu Driven Implementation
    
    # str() returns the string version of the variable "choice"
    if choice == str(1): 
        print("Clean Empty Cells - Empty cells can potentially give you a wrong result when you analyze data.")
        clean_empty_cells() # Clean Empty Cells
        
    elif choice == str(2):
        print("Clean Data of Wrong Format - Cells with data of wrong format can make it difficult, or even impossible, to analyze data. To fix it, you have two options:-")
        clean_data_wrong_format() # Clean Data of Wrong Format
        
    elif choice == str(3):
        fix_wrong_data() # Fix Wrong Data
        
    elif choice == str(4):
        print("Duplicate rows are rows that have been registered more than one time.")
        remove_duplicates() # Remove Duplicates
        
    elif choice == str(5): 
        break # Exit loop
        
    else:
        print("Error: Invalid Input! Please try again.")
