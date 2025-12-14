# import part
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch

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
