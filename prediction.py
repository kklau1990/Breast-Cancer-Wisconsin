import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('breast-cancer-wisconsin-data.csv')

# look at header
df.head()

# investigate variable property
df.info()
df.dtypes

# get total missing value per variable
df.isnull().sum()

# not sure why the dataframe is creating a spurious column, drop it
df = df.drop('id', 1)
df = df.drop('Unnamed: 32', 1)

# encode diagnosis (target variable)
le = preprocessing.LabelEncoder()
le.fit(df['diagnosis'])
le.classes_
df['diagnosis'] = le.transform(df['diagnosis'])
df.dtypes


# split into 3 histograms to view all variables
# plot histogram
f, axes = plt.subplots(2, 6, figsize=(10, 15), sharex=False)
idx = 0
for r in range(2):
    for c in range(6):
        sb.distplot(df.iloc[:, idx], color="skyblue", ax=axes[r, c], axlabel=df.columns[idx])
        idx += 1

f, axes = plt.subplots(2, 6, figsize=(10, 15), sharex=False)
for r in range(2):
    for c in range(6):
        sb.distplot(df.iloc[:, idx], color="skyblue", ax=axes[r, c], axlabel=df.columns[idx])
        idx += 1

f, axes = plt.subplots(2, 3, figsize=(10, 15), sharex=False)
for r in range(2):
    for c in range(3):
        sb.distplot(df.iloc[:, idx], color="skyblue", ax=axes[r, c], axlabel=df.columns[idx])
        idx += 1

f, axes = plt.subplots(1, 1, figsize=(10, 15), sharex=False)
sb.distplot(df.iloc[:, idx], color="skyblue", axlabel=df.columns[idx])

# Look at outliers of radius_mean and texture_mean
melted_data = pd.melt(df, id_vars='diagnosis', value_vars=['radius_mean', 'texture_mean'])
plt.figure(figsize=(15, 10))
sb.boxplot(x='variable', y='value', hue='diagnosis', data=melted_data)
plt.show()

df.describe()

# Heatmap generation to view variable correlations
f,ax = plt.subplots(figsize=(10, 10))
sb.heatmap(df.corr(), annot=True, linewidths=.1, fmt='.1f', ax=ax)

# Feature selection based on covariance effect and correlation strength
# radius_mean & perimeter_mean & area_mean are highly correlated
# compactness mean & concavity mean & concave points_mean are highly correlated
# radius_se & perimeter_se & area_se are highly correlated
# radius_worst & perimeter_worst & area_worst are highly correlated
# compactness_worst & concavity_worst & concave points_worst & actual_dimension_worst are highly correlated
# We will select only 1 of the 3 options above with highest correlation strength
# For variables that not found from above lists, at least 0.5 / - 0.5 correlation strength is considered significant
# After elimination, selected variables are:
# radius_mean, concave points_mean, radius_se, radius_worst, texture_worst, concave points_worst

df = df[['radius_mean', 'concave points_mean', 'radius_se', 'radius_worst', 'texture_worst',
        'concave points_worst', 'diagnosis']]

# Heatmap generation to view variable correlations
f,ax = plt.subplots(figsize=(7, 7))
sb.heatmap(df.corr(), annot=True, linewidths=.1, fmt='.1f', ax=ax)

# Data partitioning
# ensuring we get consistent result
seed = 0
x = df.iloc[:, df.columns != 'diagnosis'].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

result_table = pd.DataFrame(columns=['classifier', 'tpr', 'fnr', 'acc'])

classifier = [LogisticRegression(random_state=seed),
              DecisionTreeClassifier(random_state=seed),
              RandomForestClassifier(random_state=seed)]
models = []
for cls in classifier:
    model = cls.fit(X_train, y_train)
    models.append(model)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(cls.__class__.__name__ + ' Confusion Matrix \n')
    print(cm)
    tpr, fnr = cm[0][0], cm[1][1]
    result_table = result_table.append({'classifier': cls.__class__.__name__,
                                       'tpr': tpr,
                                       'fnr': fnr,
                                       'acc': accuracy}, ignore_index=True)

model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=50, shuffle=False)
models.append(model)
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Keras Deep Learning Confusion Matrix \n')
print(cm)
tpr, fnr = cm[0][0], cm[1][1]
result_table = result_table.append({'classifier': 'keras deep learning',
                                   'tpr': tpr,
                                   'fnr': fnr,
                                   'acc': accuracy}, ignore_index=True)
result_table.set_index('classifier', inplace=True)

# Optional for testing
dummy_x = [[12.00, 1.09, 0.0003, 19.23, 30, 0.07]]
dummy_y = models[2].predict(dummy_x)
print('Predicted diagnosis is: {}'.format(le.inverse_transform(dummy_y)[0]))

# All work above reference from https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
