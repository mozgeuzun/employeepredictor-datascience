#This code uses a random forest classifier to train and test a model on the employee data

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Read the dataset
df = pd.read_csv('employee_after_cleaning.csv')
"""
# Calculate the correlation between all pairs of features
corr= df.corr()
print(corr)

# Plot a heatmap of the correlation matrix
sns.heatmap(corr, annot=True)
# plt.show()
"""



# Iterate through each column in the dataframe
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

#Reorder the columns so that the target column is first
df = df.iloc[:, [11,1,2,3,4,5,6,7,8,10,12]]
print(df)

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 1:df.shape[1]].values 
Y = df.iloc[:, 0].values 

# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

#Use Random Forest Classification algorithm
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
forest.fit(X_train, Y_train)
#Get the accuracy on the training data
forest.score(X_train, Y_train)
print(forest.score(X_train, Y_train))
 
#Show the confusion matrix and accuracy for  the model on the test data
#Classification accuracy is the ratio of correct predictions to total predictions made.
cm = confusion_matrix(Y_test, forest.predict(X_test))

# Calculate true negatives, true positives, false negatives, and false positives
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

# Print the confusion matrix and the model's accuracy on the test data
print(cm)
print('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
print()
# Calculate the feature importances and sort them in descending order
importances = pd.DataFrame({'feature':df.iloc[:, 1:df.shape[1]].columns,'importance':np.round(forest.feature_importances_,3)}) 
importances = importances.sort_values('importance',ascending=False).set_index('feature')

#Print importance values and plot
print(importances)
importances.plot.bar()
plt.show()


# Select the features and target columns
X = df[['age', 'gender_full', 'BUSINESS_UNIT','STATUS_YEAR','termreason_desc']]
Y = df['STATUS']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Oversample the minority class (status=1) to balance the classes
X_train_resampled, Y_train_resampled = SMOTE(random_state=2).fit_resample(X_train, Y_train)
# Print the value counts of the STATUS column after oversampling
print(Y_train_resampled.value_counts())
# Create a countplot of the Y_train_resampled values
sns.countplot(x=Y_train_resampled)

# Show the plot
plt.show()
# Create the logistic regression model
logreg = LogisticRegression(solver='liblinear', max_iter=1000)

# fit the model to the training data
logreg.fit(X_train_resampled, Y_train_resampled)

# now you can use the model to make predictions
# suppose you want to predict the status for a person with age 30, gender_full 'male', and BUSINESS_UNIT stores
age = 30
gender_full = 1
BUSINESS_UNIT = 0
STATUS_YEAR = 2010
termreason_desc = 1

# create a dataframe with the input data
input_data = pd.DataFrame({'age': [age], 'gender_full': [gender_full], 'BUSINESS_UNIT': [BUSINESS_UNIT], 'STATUS_YEAR': [STATUS_YEAR], 'termreason_desc': [termreason_desc]})

# use the predict method to make a prediction
prediction = logreg.predict(input_data)[0]

print(f"The predicted status for a person with age {age}, gender (0=female,1=male){gender_full}, and business unit (0=headoffice,1=stores){BUSINESS_UNIT} is:")
print(prediction)
if prediction == 0:
    print("Employee will be terminated")
elif prediction == 1:
    print("Employee will be retained")

# Calculate the accuracy of the model
accuracy = logreg.score(X_test, Y_test)
print("Accuracy of logistic regression model: {:.2f}".format(accuracy))





"""
# 5 fold cross VALIDATION
X = df.iloc[:, 1:df.shape[1]].values
Y = df.iloc[:, 0].values
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, random_state=2)

# Create the model with regularization
classifier = svm.LinearSVC(penalty='l1', dual=False, C=1)
# Use cross_val_score to evaluate the model using 5-fold cross-validation


classifier.fit(X_train,Y_train)

# Make predictions on the training data
X_train_prediction = classifier.predict(X_train)

# Calculate the accuracy of the model on the training data
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy on train data:", training_data_accuracy)

# Make predictions on the test data
X_test_prediction = classifier.predict(X_test)

# Calculate the accuracy of the model on the test data
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data:", testing_data_accuracy)

"""