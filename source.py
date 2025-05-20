# fake_news_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Sample dataset (you can replace this with a larger one)
def load_sample_data():
    data = {
        'text': [
            'Breaking news: Government launches new education policy.',
            'Scientists discovered a new element in the periodic table.',
            'Click here to win a free iPhone now!',
            'Obama arrested for stealing pizza from the White House.',
            'Local man saves kitten stuck in tree.',
            'COVID-19 vaccine causes mind control, experts say!',
            'The earth is flat and NASA is hiding the truth.',
            'New study shows benefits of drinking water.',
            'Aliens have landed in Nevada desert, officials confirm.',
            'Stocks are rising due to positive economic data.'
        ],
        'label': [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # 0 = Real, 1 = Fake
    }
    return pd.DataFrame(data)

# Step 2: Preprocessing and model training
def train_fake_news_model(df):
    X = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n🧪 Model Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, tfidf

# Step 3: CLI prediction interface
def predict_news(model, vectorizer):
    print("\n🔎 Fake News Detector - Type your headline/text below")
    while True:
        user_input = input("\nEnter news text (or 'exit' to quit):\n> ")
        if user_input.lower() == 'exit':
            break
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        print("🧾 Result:", "FAKE" if prediction == 1 else "REAL")

# Step 4: Main
def main():
    print("📰 Exposing the Truth with Advanced Fake News Detection (NLP)")
    df = load_sample_data()
    print("\n📊 Sample Dataset:\n", df)

    model, vectorizer = train_fake_news_model(df)
    predict_news(model, vectorizer)

# Corrected the typo from name to _name_
if _name_ == "_main_":
    main()
