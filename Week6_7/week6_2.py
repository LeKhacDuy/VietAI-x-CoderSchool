import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('diabetes_data.csv')
# profile = ProfileReport(data,title = "diabetes")
# profile.to_file("diabetes.html")
#tuong quan cao

label_encoders = {}
for columns in data:
    label = LabelEncoder()
    data[columns] = label.fit_transform(data[columns].astype(str)) 
    label_encoders[columns] = label  

data = data.drop(["Gender"],axis=1)
data['CombinedTarget'] = (
    data['Obesity'] * 2 + data['DiabeticClass']
) 
x = data.drop(['Obesity', 'DiabeticClass', 'CombinedTarget'], axis=1)
y = data['CombinedTarget']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

standard = StandardScaler()

x_train = standard.fit_transform(x_train)
x_test = standard.transform(x_test)

model = RandomForestClassifier(random_state=100)
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(classification_report(y_test,y_predict))



