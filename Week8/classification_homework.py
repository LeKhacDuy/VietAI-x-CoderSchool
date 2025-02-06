import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def filter_location(location):
    temp = location.split(",")
    if len(temp) > 1:
        return temp[1].strip()
    else:
        return location


data = pd.read_excel("job_dataset.ods", engine="odf", dtype="str")
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1, inplace=False)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25, stratify=y)

# Preprocess
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
# processed_data = vectorizer.fit_transform(x_train["title"])
# vectorizer.vocabulary_ returns a dictionary of index and unique words
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(processed_data.shape)


'''
Các cách để cải thiện mô hình:
1. Dùng param handle_unknown = "ignore" trong OneHotEncoder để xử lý các sample có trong tập test mà không xuất hiện
trong tập train
2. Cải thiện tốc độ chạy mô hình: dùng parameters max_df và min_df để giảm số lượng Feature trong Tfidf Vector. Chỉ giữ
lại các token có frequency trong range [0.01, 0.95]
'''
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=0.95, min_df=0.01), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.99, min_df=0.01), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=0.99, min_df=0.01), "industry")
], remainder="passthrough")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=2025)),
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
class_report = classification_report(y_test, y_pred)
print(class_report)

