# Yippee
#Removing teacher quality and previous scores from the dataset. 
#Feature engineering, combining features like grade/hour studied.
#Categorical variables with natural ordering, like low-med-high are encoded into ordinal variables, 1, 2, 3. 
#Boolean values are converted into 0 1 for yes no.
#Other features such as school type are one hot encoded.


#Import libraries
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming file is in the same directory, import data.
dataf = pd.read_csv("StudentPerformanceFactors.csv")

#Remove Teacher quality, previous scores, and Distance from home, since they are either subjective or not useful.
#[Teacher_Quality], [Previous_Scores], [Distance_from_Home]
dataf = dataf.drop(labels=["Teacher_Quality","Previous_Scores","Distance_from_Home", "Peer_Influence"], axis=1)

# Ordinal Label Encoding Low --> 1, Medium --> 2, High --> 3
mapper = {"Low":1, "Medium":2, "High":3}
dataf["Parental_Involvement"] = dataf["Parental_Involvement"].map(mapper)
dataf["Access_to_Resources"] = dataf["Access_to_Resources"].map(mapper)
dataf["Family_Income"] = dataf["Family_Income"].map(mapper)
dataf["Motivation_Level"] = dataf["Motivation_Level"].map(mapper)

#Encode Boolean values into binary currently 1 for yes, 0 for no. Can change depending on how we want it.
mapper = {"Yes":1, "No":0}
dataf["Extracurricular_Activities"] = dataf["Extracurricular_Activities"].map(mapper)
dataf["Internet_Access"] = dataf["Internet_Access"].map(mapper)
dataf["Learning_Disabilities"] = dataf["Learning_Disabilities"].map(mapper)

# One-hot encoding, for School_Type, Peer_Incluence, Parental_Education_Level. True = 1, False = 0.
columns_to_dummy = ['Parental_Education_Level', 'School_Type', 'Gender']
dataf = pd.get_dummies(dataf, columns=columns_to_dummy)

# Custom interaction terms
dataf['Study_Per_Motivation'] = dataf['Hours_Studied'] * dataf['Motivation_Level']
dataf['Attendance_Per_Motivation'] = dataf['Attendance'] * dataf['Motivation_Level']
dataf['Study_Per_Attendance'] = dataf['Hours_Studied'] * dataf['Attendance']

dataf['Study_Per_Parental_Involvement'] = dataf['Hours_Studied'] * dataf['Parental_Involvement']
dataf['Attendance_Per_Parental_Involvement'] = dataf['Attendance'] * dataf['Parental_Involvement']

dataf['Study_Per_Resource_Access'] = dataf['Hours_Studied'] * dataf['Access_to_Resources']

#Normalization
# Assuming 'dataf' is your DataFrame
scaler = MinMaxScaler()


# Apply normalization (scaling between 0 and 1) to all columns
dataf_normalized = scaler.fit_transform(dataf)

# Convert back to a DataFrame
dataf = pd.DataFrame(dataf_normalized, columns=dataf.columns)

# Splitting the data into 60/20/20 for train/validate/test.
# Split into training (60%) and remaining (40%)
train_data, temp_data = train_test_split(dataf, test_size=0.4, random_state=42)

# Split remaining 40% into validation (20%) and test (20%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Separate Target Variable
X = dataf.drop(columns=['Exam_Score'])
y = dataf['Exam_Score']

X_train = train_data.drop(columns=['Exam_Score'])
y_train = train_data['Exam_Score']

X_val = val_data.drop(columns=['Exam_Score'])
y_val = val_data['Exam_Score']

X_test = test_data.drop(columns=['Exam_Score'])
y_test = test_data['Exam_Score']

dataf