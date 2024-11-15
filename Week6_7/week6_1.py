import pandas as pd
from ydata_profiling import ProfileReport
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

data = pd.read_csv('car.csv')

data['mpg_class'] = (data['mpg'] > 23).astype(int)

data = data.drop(['mpg', 'car name','horsepower'], axis=1)
x = data.drop('mpg_class', axis=1)
y = data['mpg_class']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(random_state=100)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))

