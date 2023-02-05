# This code is for the decide the feature selection

import pandas as pd


# Read the dataset into a Pandas DataFrame
df = pd.read_csv('employee_data.csv')
# Print all of the data types and their unique values
for column in df.columns:
    if df[column].dtype == object:
        print((str(column)) + " : " + str(df[column].unique()))
        print(df[column].value_counts())
        print("----------------------------------------------------------")



# Print the unique values in the 'termreason_desc' column
# unique_values = df['termreason_desc'].unique()
# print(unique_values)

#Print the unique values in the 'termtype_desc' column
# unique_values = df['termtype_desc'].unique()
# print(unique_values)

#Print the unique values in the 'department_name' column
# unique_values = df['department_name'].unique()
# print(unique_values)

#Print the unique values in the 'job_title' column
# unique_values = df['job_title'].unique()
# print(unique_values)

#Print the unique values in the 'city_name' column
# unique_values = df['city_name'].unique()
# print(unique_values)
