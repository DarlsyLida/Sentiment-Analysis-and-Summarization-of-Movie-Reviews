import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load spaCy for text processing
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('vader_lexicon')


# Initialize VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to predict sentiment using VADER
def predict_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    if sentiment_score['compound'] >= 0:
        return "Positive"
    elif sentiment_score['compound'] <= 0:
        return "Negative"
    else:
        return "None"

# Visualize sentiment distribution with a pie chart
def visualize_sentiment(sentiment):
    show_plot = False 
    sentiment_data = {'Positive': sentiment.count('Positive'), 'Negative': sentiment.count('Negative'), }
    df = pd.DataFrame(list(sentiment_data.items()), columns=['Sentiment', 'Count'])
    fig = px.pie(df, names='Sentiment', values='Count', title='Sentiment Distribution')
    st.plotly_chart(fig)

# Generate Wordcloud from the input text
# Generate Wordcloud from the input text
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Create a figure and axis explicitly
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')  # Hide axes
    
    # Now use st.pyplot with the figure object
    st.pyplot(fig)


# Preprocess the input text: Remove stopwords and non-alphabetical tokens
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    doc = nlp(text.lower())
    processed_text = " ".join([token.text for token in doc if token.text not in stop_words and token.is_alpha])
    return processed_text

# Extract aspects (topics) using LDA
def extract_aspects(text, n_topics=3):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Vectorize the text for LDA
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform([processed_text])

    # Apply LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Extract topics (aspects) based on LDA
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [words[i] for i in topic.argsort()[:-6 - 1:-1]]  # top 5 words per topic
        topics.append(" ".join(topic_words))
    return topics

# Aspect-Based Sentiment Analysis (For each aspect, predict sentiment)
def aspect_based_sentiment(text, n_topics=3):
    aspects = extract_aspects(text, n_topics)
    aspect_sentiment = {}
    for aspect in aspects:
        sentiment = predict_sentiment(aspect)
        aspect_sentiment[aspect] = sentiment
    return aspect_sentiment

# Streamlit UI Setup
st.title("Sentiment Analysis, Aspect-Based Sentiment, and Summarization")

# Input text area for user
text_input = st.text_area("Enter text for analysis:", "")

if st.button('Analyze'):
    if text_input:
        st.write("Analyzing...")

        # Sentiment Analysis (predict sentiment for whole text using VADER)
        sentiment = predict_sentiment(text_input)
        st.write(f"Overall Sentiment: {sentiment}")

        # Aspect-Based Sentiment Analysis using LDA
        aspect_sentiments = aspect_based_sentiment(text_input)
        st.write("Aspect-Based Sentiment Analysis:", aspect_sentiments)

        # Visualize sentiment distribution
        visualize_sentiment([sentiment])

        # Wordcloud visualization of the input text
        generate_wordcloud(text_input)
