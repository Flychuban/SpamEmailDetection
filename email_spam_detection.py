from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('./data/spam_ham_dataset.csv', usecols=['text', 'label'])
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Analyzing the data
# df.info()
# df.describe()
# df.isna().sum()
# ax, fig = plt.subplots()
# df['label'].value_counts().plot(kind='bar')
# plt.show()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=10)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(df['text'], df['label'])

# Evaluate the model
acc = accuracy_score(y_test, pipeline.predict(X_test))