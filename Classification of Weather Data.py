# -*- coding: utf-8 -*-
"""
Created on Mon July 10   16:36:10 2018

@author: Navneet
"""
#Importing the Necessary Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Creating a Pandas DataFrame from a CSV file
data = pd.read_csv('')  #add path to your file
data.columns
data
data[data.isnull().any(axis=1)]

#Data Cleaning Steps
del data['number']

#Now let's drop null values using the *pandas dropna* function.
before_rows = data.shape[0]
print(before_rows)

data = data.dropna()

after_rows = data.shape[0]
print(after_rows)

#How many rows dropped due to cleaning?
before_rows - after_rows

#Convert to a Classification Task 
#Binarize the relative_humidity_3pm to 0 or 1.
clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
print(clean_data['high_humidity_label'])


#Target is stored in 'y'.
y=clean_data[['high_humidity_label']].copy()
#y
clean_data['relative_humidity_3pm'].head()
y.head()

#Use 9am Sensor Signals as Features to Predict Humidity at 3pm
morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']
X = clean_data[morning_features].copy()
X.columns
y.columns

#Perform Test and Train split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
#type(X_train)
#type(X_test)
#type(y_train)
#type(y_test)
#X_train.head()
#y_train.describe()

#Fit on Train Set 
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)
type(humidity_classifier)

#Predict on Test Set 
predictions = humidity_classifier.predict(X_test)
predictions[:10]
y_test['high_humidity_label'][:10]

#Measure Accuracy of the Classifier 
accuracy_score(y_true = y_test, y_pred = predictions)

