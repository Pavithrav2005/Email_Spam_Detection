import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model not found. Please run `train_model.py` first.")
        return None

def main():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    model = load_model()

    if model is None:
        return

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Email & SMS Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter a message to check if it's spam or not.</p>", unsafe_allow_html=True)

    input_sms = st.text_area("Enter your message here:", height=150)

    if st.button('Analyze Message'):
        if input_sms:
            transformed_sms = transform_text(input_sms)
            num_characters = len(input_sms)
            num_words = len(nltk.word_tokenize(input_sms))
            num_sentences = len(nltk.sent_tokenize(input_sms))
            sentiment = sid.polarity_scores(input_sms)['compound']

            input_df = pd.DataFrame({
                'transformed_text': [transformed_sms],
                'num_characters': [num_characters],
                'num_words': [num_words],
                'num_sentences': [num_sentences],
                'sentiment': [sentiment]
            })

            result = model.predict(input_df)[0]

            if result == 1:
                st.error("🚨 This looks like SPAM!")
            else:
                st.success("✅ This seems like a legitimate message.")
        else:
            st.warning("Please enter a message to analyze.")

if __name__ == "__main__":
    main()