import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris['frame']

# Display the first few rows
print(df.head())

# Check the structure of the dataset
print(df.info())

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# No missing values in Iris dataset, but if there were, here's how to handle them:
# df = df.fillna(df.mean())  # Filling missing values with mean
# df = df.dropna()           # Dropping rows with missing values (choose one approach)

# Basic statistics
print(df.describe())


# Group by species and calculate mean for numerical columns
grouped = df.groupby('target').mean()
print(grouped)

# Map target to species names for better readability
df['species'] = df['target'].map(lambda x: iris['target_names'][x])
print(df.groupby('species').mean())


import matplotlib.pyplot as plt
import seaborn as sns


# Line chart for trends (assuming sorted data or time-related data is not available in Iris, so creating one)
df['index'] = range(len(df))  # Create an index to simulate time
plt.figure(figsize=(10, 6))
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.plot(df['index'], df['sepal width (cm)'], label='Sepal Width')
plt.title("Sepal Dimensions Over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()


# Bar chart for average petal length by species
species_mean = df.groupby('species')['petal length (cm)'].mean()
species_mean.plot(kind='bar', color='skyblue', figsize=(8, 6))
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()


# Histogram for petal length distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['petal length (cm)'], kde=True, color='purple')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()


# Scatter plot for sepal length vs. petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='bright')
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

try:
    # Attempt to load a dataset
    iris = load_iris(as_frame=True)
    df = iris['frame']
except Exception as e:
    print(f"An error occurred: {e}")
