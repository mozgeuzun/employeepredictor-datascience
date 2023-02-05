#This code is the first process called data pre-processing
import pandas as pd

# Read the dataset into a Pandas DataFrame
df = pd.read_csv('employee_data.csv')
print(df.dtypes)

# MISSING VALUE 
# Identify the columns with missing values
df.isnull().sum()
print("isnull")
print(df.isnull().sum())
# Handle missing values by dropping rows with missing values
df.dropna(inplace=True)
# Handle missing values by imputing the mean value
df.fillna(df.mean(numeric_only=True), inplace=True)
# Print the data after handling missing values
print("\n Print the data after handling missing values")
print(df)


# NOISY DATA
# Remove duplicates
df.drop_duplicates(inplace=True)
# Print the data after removing duplicates
print("\n Print the data after removing duplicates")
print(df)
# Drop rows with missing values
df = df.dropna(axis=0, how='any')
# Convert non-numeric columns to numeric
df = df.apply(pd.to_numeric, errors='ignore')
# Print the data after handling noisy data
print("\n Print the data after handling noisy data")
print(df)


# OUTLIERS VALUE
# Remove outliers using the z-score method
for col in df.select_dtypes(include=['float', 'int']).columns:
    mean = df[col].mean()
    std = df[col].std()
    z_score = (df[col] - mean) / std
    print(z_score)
    z_score = z_score.abs() > 3
    df = df[~z_score]
# Print the data after removing outliers
print("\nPrint the data after removing outliers")
print(df)


#INCONSISTENT DATA
# Handle inconsistent data by replacing invalid values with the most common value
#for col in df.columns:
#   # Get the most common value
#  most_common = df[col].value_counts().index[0]
#  # Replace invalid values with the most common value
#    df[col].replace(to_replace=r'^[^\d]+$', value=most_common, regex=True, inplace=True)
# Print the data after handling inconsistent data
#print("\nPrint the data after handling inconsistent data")
#print(df['job_title'])




# Convert the date fields to numerical values
df["recorddate_key"] = pd.to_datetime(df["recorddate_key"]).apply(lambda x: (x - pd.to_datetime("1970-01-01")).days)
df["birthdate_key"] = pd.to_datetime(df["birthdate_key"]).apply(lambda x: (x - pd.to_datetime("1970-01-01")).days)
df["orighiredate_key"] = pd.to_datetime(df["orighiredate_key"]).apply(lambda x: (x - pd.to_datetime("1970-01-01")).days)
df["terminationdate_key"] = pd.to_datetime(df["terminationdate_key"]).apply(lambda x: (x - pd.to_datetime("1970-01-01")).days)
print("\nPrint the data after handling date fields to numerical values")
print(df)

# LABEL ENCODING
df.replace({"STATUS":{'TERMINATED':0, 'ACTIVE':1}},inplace=True)
df.replace({"BUSINESS_UNIT":{'HEADOFFICE':0, 'STORES':1}},inplace=True)
df.replace({"gender_full":{'Female':0, 'Male':1}},inplace=True)
df.replace({"termreason_desc":{'Not Applicable':0, 'Retirement':1, 'Resignaton':2, 'Layoff':3}}, inplace=True)
df.replace({"termtype_desc":{'Not Applicable':0, 'Voluntary':1, 'Involuntary':2}}, inplace=True)
print("\nPrint the data after label encoding")
print(df)


# Dropping features after running unique_values.py (FEATURE SELECTION)
df = df.drop('recorddate_key',axis=1)
df = df.drop('birthdate_key',axis=1)
df = df.drop('orighiredate_key',axis=1)
df = df.drop('terminationdate_key',axis=1)
df = df.drop('gender_short',axis=1)
df = df.drop('EmployeeID',axis=1)
print("\nPrint the data after dropping")
print(df)


# Copy this data to a new file after pre processing
df.to_csv("employee_after_cleaning.csv", index=True)
