import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import Speller
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# Load data
train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")

# Initialize NLP tools
stemmer = PorterStemmer()
spell = Speller()
stop_words = set(stopwords.words('english'))

# Define preprocessing function
def advanced_preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = spell(text)
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Parallel preprocessing
train_df['cleaned_text'] = Parallel(n_jobs=-1)(delayed(advanced_preprocess_text)(text) for text in train_df['text'])
test_df['cleaned_text'] = Parallel(n_jobs=-1)(delayed(advanced_preprocess_text)(text) for text in test_df['text'])
print(train_df[['text', 'cleaned_text']].head())

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_text']).toarray()
X_test_tfidf = vectorizer.transform(test_df['cleaned_text']).toarray()
y_train = train_df['target']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Ensemble model
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model)], voting='soft')

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# Validate model
y_val_pred = ensemble_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Predict on test data
test_df['target'] = ensemble_model.predict(X_test_tfidf)

# Save submission
submission = test_df[['id', 'target']]
submission.to_csv('submission3.csv', index=False)
print(submission.head())