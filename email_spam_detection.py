from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pickle

def load_data():
    df = pd.read_csv('./data/spam_ham_dataset.csv', usecols=['text', 'label'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def clean_data(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '')
    return df

def analyze_data(df):
    df.info()
    df.describe()
    df.isna().sum()
    ax, fig = plt.subplots()
    df['label'].value_counts().plot(kind='bar')
    plt.show()
    
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    return acc

def save_model(pipeline):
    pickle.dump(pipeline, open('./models/spam_classification.pkl', 'wb'))


def make_prediction(pipeline, text):
    prediction = pipeline.predict([text])[0]
    if prediction == 1:
        print('This is a spam email')
    else:
        print('This is NOT a spam email')

def build_pipeline():
    df = load_data()
    df = clean_data(df)
    analyze_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = train_model(X_train, y_train)
    acc = evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline)
    return pipeline

if __name__ == '__main__':
    pipeline = build_pipeline()
    make_prediction(pipeline, 'Hi Jude, Do you want to play football this weekend?')