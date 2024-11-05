import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt


import nltk
nltk.download('stopwords')
print(stopwords.words('english'))



data=pd.read_csv(r"train.csv")
data.head(10)


test_data=pd.read_csv(r"test.csv")
test_data


data[data["category"]=="Online and Social Media Related Crime"].sub_category.value_counts()

data=pd.concat([data,test_data],axis=0)
data


data.isnull().sum()

data.category.value_counts()

data[data["category"] == "Online Financial Fraud"].sub_category.value_counts()

data.shape

data=data.dropna()
data.shape

import re
port_stem = PorterStemmer()



def clean_data(combine):
    stemmed_content = re.sub('[^a-zA-Z)]',' ', combine)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content



data['crimeaditionalinfo'] = data['crimeaditionalinfo'].apply(clean_data)
data
clean_data_from_data=data
import seaborn as sns

sns.countplot(clean_data_from_data, x="category")
plt.title("category")
plt.show()



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
def Encoder(data,i):
    data[i]=le.fit_transform(data[i])
    return data

data=Encoder(data,"category")
data


data.category.unique()
values=list(le.inverse_transform(data["category"].unique()))
values


def value_assign(data,col):
    values=list(le.inverse_transform(data["category"].unique()))
    index=list(data[col].unique())
    d={}
    for i in range(0,len(index)):
        d[index[i]]=values[i]
    return d

d=value_assign(data,"category")
d


X=data["crimeaditionalinfo"]
Y=data["category"]
X.shape




from imblearn.over_sampling import RandomOverSampler
X_reshaped = pd.DataFrame(X).values.reshape(-1, 1)

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=100)

X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, Y)

X_resampled = pd.DataFrame(X_resampled, columns=['crimeaditionalinfo'])

print(X_resampled)
print(X_resampled.shape, y_resampled.shape)


sns.countplot(y_resampled.value_counts())
plt.title("category")
plt.show()
print(y_resampled.value_counts())



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


tfv = TfidfVectorizer(max_features=100)
X=tfv.fit_transform(X_resampled['crimeaditionalinfo']).toarray()
X.shape


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y_resampled,test_size=0.4,random_state=1000)

x_train.shape,y_train.shape



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)


RFC.score(x_test,y_test)
import joblib
joblib.dump(RFC,"RFC.sav")


y_prediction=RFC.predict(x_test)
y_prediction


print(confusion_matrix(y_test,y_prediction))

print(classification_report(y_test,y_prediction))

from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)

DT.score(x_test,y_test)
joblib.dump(DT,"DT.sav")

data0=data[data["category"]==0]
data1=data[data["category"]==1]
data2=data[data["category"]==2]
data3=data[data["category"]==3]
data4=data[data["category"]==4]
data5=data[data["category"]==5]
data6=data[data["category"]==6]
data7=data[data["category"]==7]
data8=data[data["category"]==8]
data9=data[data["category"]==9]
data10=data[data["category"]==10]
data11=data[data["category"]==11]

i=0
def model_process(data_sub):
    data_sub = data_sub.drop("category", axis=1)
    data_sub = Encoder(data_sub, "sub_category")
    if (len(data_sub) == 1):
        data_sub = pd.concat([data_sub, data_sub, data_sub, data_sub, data_sub, data_sub, data_sub], axis=0)
    X = data_sub["crimeaditionalinfo"]
    Y = data_sub["sub_category"]
    sns.countplot(Y.value_counts())
    plt.title(d[i])
    plt.show()
    print("Unique classes in Y:", Y.unique())

    if Y.nunique() <= 1:
        tfv = TfidfVectorizer(max_features=100)
        X = tfv.fit_transform(X).toarray()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        RFC = RandomForestClassifier()
        RFC.fit(x_train, y_train)
        print(RFC.score(x_test, y_test))
        return RFC
    from imblearn.over_sampling import RandomOverSampler
    X_reshaped = pd.DataFrame(X).values.reshape(-1, 1)

    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=100)

    X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, Y)

    X_resampled = pd.DataFrame(X_resampled, columns=['crimeaditionalinfo'])
    sns.countplot(y_resampled.value_counts())
    plt.title(d[i])
    plt.show()
    tfv = TfidfVectorizer(max_features=100)
    X = tfv.fit_transform(X_resampled['crimeaditionalinfo']).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, y_resampled, test_size=0.2)
    RFC = RandomForestClassifier()
    RFC.fit(x_train, y_train)
    print(RFC.score(x_test, y_test))
    return RFC


model0=model_process(data0)
i+=1
model1=model_process(data1)
i+=1
model2=model_process(data2)
i+=1
model3=model_process(data3)
i+=1
model4=model_process(data4)
i+=1
model5=model_process(data5)
i+=1
model6=model_process(data6)
i+=1
model7=model_process(data7)
i+=1
model8=model_process(data8)
i+=1
model9=model_process(data9)
i+=1
model10=model_process(data10)
i+=1
model11=model_process(data11)

import joblib

joblib.dump(model0,"model0.sav")
joblib.dump(model1,"model1.sav")
joblib.dump(model2,"model2.sav")
joblib.dump(model3,"model3.sav")
joblib.dump(model4,"model4.sav")
joblib.dump(model5,"model5.sav")
joblib.dump(model6,"model6.sav")
joblib.dump(model7,"model7.sav")
joblib.dump(model8,"model8.sav")
joblib.dump(model9,"model9.sav")
joblib.dump(model10,"model10.sav")
joblib.dump(model11,"model11.sav")






