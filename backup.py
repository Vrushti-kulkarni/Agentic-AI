import requests
from bs4 import BeautifulSoup
from readability import Document
from keybert import KeyBERT
from transformers import pipeline
from phi.agent import Agent
from phi.tools.youtube_tools import YouTubeTools
from phi.llm.groq import Groq
from phi.assistant import Assistant
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def extract_text_from_url(url):
    """Extract main text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        doc = Document(response.text)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""

def extract_keywords(text):
    """Extract top keywords from the text."""
    try:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=5
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def summarize_text(text, max_length=150, min_length=30):
    """Generate a summary of the text."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""

def process_url(url):
    """Process a given URL and print results."""
    text = extract_text_from_url(url)
    if text:
        summary = summarize_text(text)
        keywords = extract_keywords(text)
        
        print(f"\n🔹 **Detected Topic(s) for {url}:** {keywords}")
        print(f"\n📌 **Summary:** {summary}")
        
        # Search for YouTube videos using the extracted keywords
        search_youtube_videos(keywords)
    else:
        print(f"Could not extract content from {url}")

def search_youtube_videos(keywords):
    """Search for YouTube videos related to the given keywords."""
    # Initialize Assistant with the specified Groq model
    assistant = Assistant(
        llm=Groq(model="mixtral-8x7b-32768"),  # You can change the model if needed
        tools=[YouTubeTools()]
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("Groq API key loaded successfully.")
    else:
        print("Error: Groq API key is not loaded.")

    # Use Assistant to search for videos related to the keywords
    query = f"Search YouTube for videos explaining {keywords[0]}."
    assistant.print_response(query, markdown=True)

def main():
    urls = [
        "https://www.geeksforgeeks.org/natural-language-processing-overview/",
        # Add more URLs here if you want to process multiple
    ]
    
    for url in urls:
        process_url(url)

if __name__ == "__main__":
    main()