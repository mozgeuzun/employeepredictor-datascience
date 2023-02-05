#This code is the second process called data visualization. It compares features.

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Read the csv file
df = pd.read_csv('employee_after_cleaning.csv')

# Print the value counts for the 'STATUS' column
print(df['STATUS'].value_counts())
# Use seaborn to create a countplot for the 'STATUS' column
sns.countplot(x='STATUS', data=df)
# Show the plot
plt.show()


# Plot the count of employees in each age group, separated by their status (retained or terminated)
sns.countplot(x='age', hue='STATUS', data=df)
# Show the plot
plt.show()


# Plot the count of employees in each gender, separated by their status (active or terminated)
sns.countplot(x='gender_full', hue='STATUS', data=df)
plt.show()


# Plot the count of employees who left and stayed for each job title
sns.countplot(x='job_title', hue='STATUS', data=df)
plt.show()

# Plot the count of employees with different levels of length of service, colored by their status (active or terminated)
sns.countplot(x='length_of_service', hue='STATUS', data=df)
plt.show()

# Calculate the correlations between the features and the target column
corr = df.corr()['STATUS']


# Plot the correlations between the features and the target column using a heatmap
sns.heatmap(corr.to_frame(), annot=True)
plt.show()

# Calculate basic statistics for numerical features
print(df.describe())

# Plot histograms for all numerical features
df.hist(figsize=(10, 10))
plt.show()

# Plot scatter plots for all pairs of numerical features
sns.pairplot(df)
plt.show()

# Calculate the correlation between all pairs of features
corr= df.corr()
print(corr)

# Plot a heatmap of the correlation matrix
sns.heatmap(corr, annot=True)
plt.show()



# Select a feature and the target column
feature = 'age'
x = df[feature]
y = df['STATUS']
# Use a t-test to compare the means of the target column for different groups of the feature
t_stat, p_val = stats.ttest_ind(x[y == 1].values, x[y == 0].values)

# Print the t-statistic and p-value
print(f't-statistic: {t_stat:.3f}')
print(f'p-value: {p_val:.3f}')


