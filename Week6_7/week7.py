import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression

data = pd.read_csv("csgo.csv")
# profile = ProfileReport(data, title="Csgo Report", explorative=True)
# profile.to_file('report.html')

if 'map' in data.columns:
    label = LabelEncoder()
    data['map'] = label.fit_transform(data['map'])

data = data.drop(['team_a_rounds','team_b_rounds'],axis=1)
data = data.drop(['day','month','year','date'],axis=1)
x = data.drop(['points','result'],axis=1)
y = data['points']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
stan = StandardScaler()
x_train = stan.fit_transform(x_train)
x_test = stan.transform(x_test)
model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
mse = mean_squared_error(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

print(f"mse: {mse}")
print(f"mae: {mae}")
print(f"r2: {r2}")