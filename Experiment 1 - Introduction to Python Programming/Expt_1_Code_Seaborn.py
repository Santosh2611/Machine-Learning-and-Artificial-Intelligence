import matplotlib.pyplot as plt # Provides an implicit way of plotting
import seaborn as sb # Provides high level API to visualize data

print("\nImporting Datasets and Libraries (Importing Data as Pandas DataFrame):")
df = sb.load_dataset('flights')
print (df.head())

print("\nHistogram:")
df = sb.load_dataset('iris')
sb.histplot(df['petal_length'], kde = False) # Here, kde flag is set to False. As a result, the representation of the kernel estimation plot will be removed and only histogram is plotted.
plt.show()

print("\nPlotting Categorical Data (stripplot()):")
df = sb.load_dataset('iris')
sb.stripplot(x = "species", y = "petal_length", data = df)
plt.show()

print("\nStatistical Estimation (Bar Plot):")
df = sb.load_dataset('titanic')
sb.barplot(x = "sex", y = "survived", hue = "class", data = df)
plt.show()

print("\nStatistical Estimation (Point Plots):")
df = sb.load_dataset('titanic')
sb.pointplot(x = "sex", y = "survived", hue = "class", data = df)
plt.show()

print("\nPlotting Wide Form Data:")
df = sb.load_dataset('iris')
sb.boxplot(data = df, orient = "h")
plt.show()
