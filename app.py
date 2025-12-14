# import part
from bs4 import BeautifulSoup
import numpy as np
import torch
from googlesearch import search
import time
from docx import Document
import io


# main part
# Set page configuration
st.set_page_config(page_title="Investment Research Assistant", page_icon="📈", layout="wide")


# Title
st.title("📈 Investment Research Assistant: Stock Analysis via News & Documents")
st.markdown("Ask questions about stocks or upload documents for investment recommendations based on sentiment analysis")

st.title("📈 Investment Research Assistant: Financial Article Analysis")
st.markdown("Analyze financial articles to get investment recommendations based on sentiment analysis")

# Initialize models with caching
@st.cache_resource
@@ -32,55 +25,12 @@ def load_summarization_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)
    return summarizer


@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    return tokenizer, model


# Function: Extract text from DOCX file
def extract_text_from_docx(uploaded_file):
    """
    Extract text content from uploaded DOCX file
    """
    try:
        doc = Document(io.BytesIO(uploaded_file.read()))
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        return "".join(text_content)
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return None


# Function: Search Google for news articles
def search_google_news(query, num_results=2):
    """
    Search Google for news articles related to the query
    Returns top N URLs from search results
    """
    try:
        search_query = f"{query} stock news"
        urls = []
        
        # Get search results
        for url in search(search_query, num_results=num_results, lang="en", pause=2.0):
            urls.append(url)
            if len(urls) >= num_results:
                break
        
        return urls
    except Exception as e:
        st.error(f"Error searching Google: {str(e)}")
        return []


# Function: Enhanced Text Summarization from URL
def text_summarization(url, summarizer):
    # Add headers to mimic a browser request
@@ -116,29 +66,9 @@ def text_summarization(url, summarizer):

        return input_text
    except Exception as e:
        st.warning(f"Could not fetch article from {url}: {str(e)}")
        return None


# Function: Summarize text content
def summarize_text(text_content, summarizer):
    """
    Summarize text content directly
    """
    try:
        summary = summarizer(
            text_content,
            max_length=150,
            min_length=50,
            truncation=True
        )[0]["summary_text"]
        
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        st.error(f"Error fetching URL: {str(e)}")
        return None


# Function: Enhanced Sentiment Analysis
def analyze_sentiment(text, tokenizer, model):
    # Define label mapping
@@ -178,8 +108,7 @@ def analyze_sentiment(text, tokenizer, model):
        'all_scores': {id2label[i]: float(predictions[0][i]) for i in range(len(id2label))}
    }


# Function: Investment Research Assistant
# Function: Investiment Research Assistant
def investment_advisor(summary_text, sentiment_result):
    sentiment_label = sentiment_result['label'].lower()
    confidence = sentiment_result['score']
@@ -201,176 +130,68 @@ def investment_advisor(summary_text, sentiment_result):
        'advice': advice
    }


# Main App
def main():
    # Load models
    with st.spinner("Loading AI models..."):
        summarizer = load_summarization_model()
        sentiment_tokenizer, sentiment_model = load_sentiment_model()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["💬 Ask About Stock", "📄 Upload Document"])
    # Input section
    st.subheader("📝 Enter Financial Article URL")
    url = st.text_input("Enter the URL of the financial article:", placeholder="https://example.com/article")

    # Tab 1: Question-based analysis
    with tab1:
        st.subheader("💬 Ask About a Stock")
        user_question = st.text_input(
            "Enter your question:", 
            placeholder="Is it a good time to buy Tesla stock?"
        )
        
        if st.button("Analyze Stock", type="primary", key="analyze_stock"):
            if user_question:
                # Step 1: Search Google for news
                with st.spinner("Searching for latest news articles..."):
                    news_urls = search_google_news(user_question, num_results=2)
    if st.button("Analyze Article", type="primary"):
        if url:
            # Step 1: Summarization
            with st.spinner("Fetching and summarizing article..."):
                summary_text = text_summarization(url, summarizer)
            
            if summary_text:
                st.success("Summary generated successfully!")

                if not news_urls:
                    st.error("Could not find any news articles. Please try a different question.")
                    return
                # Display summary
                st.subheader("📄 Article Summary")
                st.write(summary_text)

                st.success(f"Found {len(news_urls)} news articles!")
                # Step 2: Sentiment Analysis
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = analyze_sentiment(summary_text, sentiment_tokenizer, sentiment_model)

                # Display found URLs
                with st.expander("📰 News Sources"):
                    for idx, url in enumerate(news_urls, 1):
                        st.write(f"{idx}. {url}")
                # Step 3: Generate Investment Advice
                result = investment_advisor(summary_text, sentiment_result)

                # Step 2: Summarize and analyze each article
                all_summaries = []
                all_sentiments = []
                # Display results
                st.markdown("---")
                st.subheader("📊 Investment Analysis Report")

                for idx, url in enumerate(news_urls, 1):
                    st.markdown(f"### Article {idx}")
                    
                    with st.spinner(f"Analyzing article {idx}..."):
                        summary_text = text_summarization(url, summarizer)
                    
                    if summary_text:
                        all_summaries.append(summary_text)
                        
                        # Display summary
                        st.write("**Summary:**")
                        st.write(summary_text)
                        
                        # Sentiment Analysis
                        sentiment_result = analyze_sentiment(summary_text, sentiment_tokenizer, sentiment_model)
                        all_sentiments.append(sentiment_result)
                        
                        # Display sentiment
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment_result['label'].upper())
                        with col2:
                            st.metric("Confidence", f"{sentiment_result['score']:.2%}")
                        
                        st.markdown("---")
                col1, col2 = st.columns(2)

                # Step 3: Overall Analysis
                if all_sentiments:
                    st.subheader("📊 Overall Investment Analysis")
                    
                    # Calculate average sentiment scores
                    avg_scores = {
                        'positive': np.mean([s['all_scores']['positive'] for s in all_sentiments]),
                        'neutral': np.mean([s['all_scores']['neutral'] for s in all_sentiments]),
                        'negative': np.mean([s['all_scores']['negative'] for s in all_sentiments])
                    }
                    
                    # Determine overall sentiment
                    overall_sentiment = max(avg_scores, key=avg_scores.get)
                    
                    # Display overall results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive", f"{avg_scores['positive']:.2%}")
                    with col2:
                        st.metric("Neutral", f"{avg_scores['neutral']:.2%}")
                    with col3:
                        st.metric("Negative", f"{avg_scores['negative']:.2%}")
                    
                    # Generate advice
                    st.subheader("💡 Investment Recommendation")
                    if overall_sentiment == 'positive':
                        st.success("✅ This stock is recommended based on recent news sentiment.")
                    elif overall_sentiment == 'negative':
                        st.error("❌ This stock is not recommended based on recent news sentiment.")
                    else:
                        st.warning("⚠️ This stock needs to adopt a wait-and-see attitude based on recent news sentiment.")
                    
            else:
                st.warning("Please enter a question about a stock.")
    
    # Tab 2: Document upload analysis
    with tab2:
        st.subheader("📄 Upload Financial Document")
        uploaded_file = st.file_uploader(
            "Upload a DOCX file containing financial information:",
            type=['docx'],
            help="Upload a Word document (.docx) for analysis"
        )
        
        if st.button("Analyze Document", type="primary", key="analyze_doc"):
            if uploaded_file is not None:
                # Step 1: Extract text from DOCX
                with st.spinner("Reading document..."):
                    text_content = extract_text_from_docx(uploaded_file)
                with col1:
                    st.metric("Sentiment", result['sentiment'].upper())

                if text_content:
                    st.success("Document loaded successfully!")
                    
                    # Display extracted text preview
                    with st.expander("📝 Document Preview"):
                        st.text_area("Content", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
                    
                    # Step 2: Summarization
                    with st.spinner("Generating summary..."):
                        summary_text = summarize_text(text_content, summarizer)
                    
                    if summary_text:
                        st.subheader("📄 Document Summary")
                        st.write(summary_text)
                        
                        # Step 3: Sentiment Analysis
                        with st.spinner("Analyzing sentiment..."):
                            sentiment_result = analyze_sentiment(summary_text, sentiment_tokenizer, sentiment_model)
                        
                        # Step 4: Generate Investment Advice
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
                        
                        # Display advice
                        st.markdown("---")
                        st.subheader("💡 Investment Recommendation")
                        if result['sentiment'] == 'positive':
                            st.success(f"✅ {result['advice']}")
                        elif result['sentiment'] == 'negative':
                            st.error(f"❌ {result['advice']}")
                        else:
                            st.warning(f"⚠️ {result['advice']}")
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")

            else:
                st.warning("Please upload a DOCX file to analyze.")

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
