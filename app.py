# import part
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch

# Set the font size with Global CSS Approach
st.markdown('''
<style>
    /* Set base font size for all text elements */
    .main-title, .subtitle, .section-header, 
    .content-text, .metric-label, .metric-value {
        font-size: 20px !important;  /* Match function title size */
    }
    
    /* If function titles use a specific class */
    .function-title {
        font-size: 20px !important;
    }
    
    /* Standardize all headers */
    h1, h2, h3, h4, h5, h6 {
        font-size: 20px !important;
    }
    
    /* Standardize paragraph text */
    p, div, span {
        font-size: 20px !important;
    }
    
    /* Standardize input labels and buttons */
    label, button, .stButton>button {
        font-size: 20px !important;
    }
</style>
''', unsafe_allow_html=True)
st.markdown("""
<style>
    /* Make the main title larger */
    h1 {
        font-size: 36px !important;
        font-weight: bold !important;
    }
    
    /* Specifically target the first heading if needed */
    .main h1:first-of-type {
        font-size: 36px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# title displayed larger
st.title("📈 Investment Research Assistant: Financial Article Analysis")
st.write("Analyze financial articles to get investment recommendations based on sentiment analysis")

# main part
# Set page configuration
st.set_page_config(page_title="Investment Research Assistant", page_icon="📈", layout="wide")

# Title
st.title("📈 Investment Research Assistant: Financial Article Analysis")
st.markdown("Analyze financial articles to get investment recommendations based on sentiment analysis")

# Initialize models with caching
@st.cache_resource
def load_summarization_model():
    # Manually load and configure the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024
    
    # Initialize pipeline with the configured tokenizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)
    return summarizer

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    return tokenizer, model

# Function: Enhanced Text Summarization from URL
def text_summarization(url, summarizer):
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        # Fetch and parse the web content with headers
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text_content = "".join([p.get_text() for p in paragraphs])
        
        if not text_content.strip():
            text_content = soup.get_text()
        
        # Generate summary
        input_text = summarizer(
            text_content,
            max_length=150,
            min_length=50,
            truncation=True
        )[0]["summary_text"]
        
        return input_text
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

# Function: Enhanced Sentiment Analysis
def analyze_sentiment(text, tokenizer, model):
    # Define label mapping
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Prepare text with context
    formatted_text = f"Generated text: {text}"
    
    # Tokenize input
    inputs = tokenizer(
        formatted_text,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    
    # Get the index of the largest output value
    max_index = np.argmax(predictions)
    
    # Convert numeric prediction to text label
    predicted_label = id2label[max_index]
    
    # Get confidence score
    confidence = predictions[0][max_index]
    
    return {
        'label': predicted_label,
        'score': float(confidence),
        'all_scores': {id2label[i]: float(predictions[0][i]) for i in range(len(id2label))}
    }

# Function: Investiment Research Assistant
def investment_advisor(summary_text, sentiment_result):
    sentiment_label = sentiment_result['label'].lower()
    confidence = sentiment_result['score']
    
    if sentiment_label == 'positive':
        advice = "This stock is recommended."
    elif sentiment_label == 'negative':
        advice = "This stock is not recommended."
    elif sentiment_label == 'neutral':
        advice = "This stock needs to adopt a wait-and-see attitude."
    else:
        advice = "Unable to determine investment recommendation."
    
    return {
        'summary': summary_text,
        'sentiment': sentiment_label,
        'confidence': confidence,
        'all_scores': sentiment_result['all_scores'],
        'advice': advice
    }

# Main App
def main():
    # Load models
    with st.spinner("Loading AI models..."):
        summarizer = load_summarization_model()
        sentiment_tokenizer, sentiment_model = load_sentiment_model()
    
    # Input section
    st.subheader("📝 Enter Financial Article URL")
    url = st.text_input("Enter the URL of the financial article:", placeholder="https://example.com/article")
    
    if st.button("Analyze Article", type="primary"):
        if url:
            # Step 1: Summarization
            with st.spinner("Fetching and summarizing article..."):
                summary_text = text_summarization(url, summarizer)
            
            if summary_text:
                st.success("Summary generated successfully!")
                
                # Display summary
                st.subheader("📄 Article Summary")
                st.write(summary_text)
                
                # Step 2: Sentiment Analysis
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = analyze_sentiment(summary_text, sentiment_tokenizer, sentiment_model)
                
                # Step 3: Generate Investment Advice
                result = investment_advisor(summary_text, sentiment_result)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Investment Analysis Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentiment", result['sentiment'].upper())
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                # Display all sentiment scores
                st.subheader("🎯 Detailed Sentiment Scores")
                score_cols = st.columns(3)
                for idx, (label, score) in enumerate(result['all_scores'].items()):
                    with score_cols[idx]:
                        st.metric(label.capitalize(), f"{score:.2%}")
                
                st.markdown("---")
                st.subheader("💡 Investment Advice")
                
                # Color-code advice based on sentiment
                if result['sentiment'] == 'positive':
                    st.success(result['advice'])
                elif result['sentiment'] == 'negative':
                    st.error(result['advice'])
                else:
                    st.warning(result['advice'])
        else:
            st.warning("Please enter a valid URL")

if __name__ == "__main__":
    main()
