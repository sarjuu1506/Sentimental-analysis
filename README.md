# Sentimental-analysis
🛍️ Strategic AI Product Purchase Advisor
AI-Driven Consumer Intelligence Platform

This platform is a high-performance Local Edge AI tool designed to provide consumers and businesses with a data-driven "Buy/No-Buy" verdict. By scraping live product feedback or ingesting CSV datasets, the engine utilizes a DistilBERT-based NLP pipeline to extract sentiment and generate strategic business recommendations.

🚀 Key Features
Automated Decision Engine: Generates a definitive Buy/No-Buy Verdict based on a calculated Consumer Confidence Score.

Hybrid Data Ingestion: Supports live Web Scraping (Amazon, Flipkart, Blogs) via BeautifulSoup and CSV Uploads for batch processing.

Strategic Recommendations: Automatically maps negative consumer feedback to specific business improvements (e.g., logistics, quality control).

Visual Analytics: Real-time generation of sentiment distribution charts and thematic word clouds.

🏗️ System Architecture
The system follows a three-layer pipeline to ensure data integrity and high-speed processing:

Ingestion Layer: Uses a session-persistent crawler with spoofed headers to bypass anti-bot filters and extract clean review DOM elements.

Inference Layer: Processes text through a DistilBERT transformer model. The model is cached in VRAM for sub-10ms response times.

Visualization Layer: Streamlit-based dashboard provides a Consumer Confidence Score and actionable business insights.

🛠️ Tech Stack
Framework: Streamlit

NLP Model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)

Scraping: BeautifulSoup4 & Requests

Analytics: Pandas,Matplotlib

Install Dependencies:
Bash
pip install streamlit pandas torch transformers beautifulsoup4 requests wordcloud matplotlib emoji

Run the App:
Bash
streamlit run app.py

Bash
streamlit run app.py
