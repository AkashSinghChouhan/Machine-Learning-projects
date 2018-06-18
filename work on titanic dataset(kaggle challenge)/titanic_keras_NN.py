import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing,model_selection
import seaborn as sns
from  sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import keras
from keras.layers import Dense, Dropout
import tensorflow as tf
#Variable	Definition	Key
#survival	Survival	0 = No, 1 = Yes
#pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#sex	Sex	
#Age	Age in years	
#sibsp	# of siblings / spouses aboard the Titanic	
#parch	# of parents / children aboard the Titanic	
#ticket	Ticket number	
#fare	Passenger fare	
#cabin	Cabin number	
#embarked	Port of Embarkation

df=pd.read_csv(".../train.csv")

df.drop(['Name'], 1 , inplace=True)
#df.drop(['Age'],1,inplace =True)
df.drop(['Embarked'],1,inplace=True)
df.convert_objects(convert_numeric=True)

# absolute numbers
print(df["Survived"].value_counts())

# percentages
print(df["Survived"].value_counts(normalize = True))

#print(df["Survived"][df["Sex"] == 'male'].value_counts())
#print(df["Survived"][df["Sex"] == 'female'].value_counts())

pd.options.mode.chained_assignment = None 
df["Sex"][df["Sex"] == "male"] = 0
df["Sex"][df["Sex"] == "female"] = 1


#FILLING THE MISSING DATA

df["Age"] =df["Age"].fillna(df["Age"].median())
#df["Embarked"] = df["Embarked"].fillna("C")
#df["Embarked"][df["Embarked"] == 'S'] = 0
#df["Embarked"][df["Embarked"] == 'C'] = 1
#df["Embarked"][df["Embarked"] == 'Q'] = 2

#corr=df.corr()#["Survived"]
#plt.figure(figsize=(10, 10))

#sns.heatmap(corr, vmax=.8, linewidths=0.01,
       #     square=True,annot=True,cmap='YlGnBu',linecolor="white")
#plt.title('Correlation between features')
#plt.show()
#print(df.corr()["Survived"])
#print(df.Cabin.nunique())
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = { }
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x= 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)

data_col= ['PassengerId','Pclass','Age','Sex','SibSp','Parch','Cabin','Ticket','Fare']
X=df[data_col]
y=df["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 4)


model = keras.Sequential()
model.add(Dense(100, input_dim =9,kernel_initializer = 'normal', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation ='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['accuracy'])
model.fit(X_train, y_train, epochs = 200, batch_size = 10, verbose =0)

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


test=pd.read_csv(".../test.csv")
print(test.shape)
test.drop(['Name'], 1 , inplace=True)

test.convert_objects(convert_numeric=True)


test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Age"] =test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("C")
test["Embarked"][test["Embarked"] == 'S'] = 0
test["Embarked"][test["Embarked"] == 'C'] = 1
test["Embarked"][test["Embarked"] == 'Q'] = 2

def fill_missing_fare(df):
    median_fare=df[(df["Pclass"] == 3) & (df["Embarked"] == 0)]["Fare"].median()
#'S'
       #print(median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df
test=fill_missing_fare(test)
test.drop(['Embarked'],1,inplace =True)





test = handle_non_numerical_data(test)


y_pred = model.predict(test)
test["Survived"] = y_pred
cols=['PassengerId','Survived']


#import csv
#with open("titanic_result.csv", "w") as my_empty_csv:
#    pass titanic_results = test[cols]

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv("titanic_sn.csv", index=False)


