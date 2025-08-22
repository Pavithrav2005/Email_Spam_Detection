import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import requests
import zipfile
import io

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt')
except LookupError:
    nltk.download('vader_lexicon')

ps = PorterStemmer()
sid = SentimentIntensityAnalyzer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def download_and_load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    print("Extracting data...")
    df = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', header=None, names=['target', 'text'])
    return df

print("Loading and cleaning data...")
df = download_and_load_data()
df.drop_duplicates(keep='first', inplace=True)

print("Preprocessing data...")
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df['transformed_text'] = df['text'].apply(transform_text)
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

X = df[['transformed_text', 'num_characters', 'num_words', 'num_sentences', 'sentiment']]
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=3000), 'transformed_text'),
        ('scaler', StandardScaler(), ['num_characters', 'num_words', 'num_sentences', 'sentiment'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("Saving model pipeline...")
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model training complete and artifacts saved.")