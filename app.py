import streamlit as st
import pandas as pd
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import torch
import re

# --------------------------------------------------
# 1. Load Models
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device
    )

@st.cache_resource(show_spinner="Loading text classifier...")
def load_text_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

sentiment_pipe = load_sentiment_model()
text_classifier = load_text_classifier()

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

TEXT_TYPES = [
    "customer complaint",
    "customer praise",
    "product specification",
    "general information"
]

# --------------------------------------------------
# 2. Helper Functions
# --------------------------------------------------
def classify_text_type(text):
    result = text_classifier(text, TEXT_TYPES)
    return result["labels"][0]

def normalize_sentiment(label, score):
    if score < 0.65:
        return "NEUTRAL"
    return label

def clean_text_blocks(texts):
    cleaned = []
    for t in texts:
        if (
            40 < len(t) < 600
            and not re.search(
                r"(cookie|privacy|terms|policy|login|signup)",
                t.lower()
            )
        ):
            cleaned.append(t.strip())
    return list(set(cleaned))

# --------------------------------------------------
# 3. Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="AI Product Feedback Analyzer", layout="wide")
st.title("📊 AI Product Feedback Analyzer")
st.caption("Smart complaint detection • sentiment analysis • business insights")

c1, c2, c3 = st.columns(3)
c1.success("💻 GPU Enabled" if torch.cuda.is_available() else "💻 CPU Mode")
c2.info("🧠 NLP: RoBERTa + Zero-Shot")
c3.info("🎯 Business Insight Engine")

st.divider()

# --------------------------------------------------
# 4. Input Method
# --------------------------------------------------
input_method = st.radio(
    "Provide review source:",
    ("Paste Product URL", "Upload Review CSV"),
    horizontal=True
)

raw_texts = []

# CSV Input
if input_method == "Upload Review CSV":
    file = st.file_uploader("Upload CSV (must contain 'text' column)", type="csv")
    if file:
        df_input = pd.read_csv(file)
        df_input.columns = df_input.columns.str.lower()
        raw_texts = df_input["text"].dropna().astype(str).tolist()

# URL Input
else:
    url = st.text_input("Paste any product / review page URL")
    if url:
        try:
            with st.spinner("🌐 Extracting webpage text..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=15)
                soup = BeautifulSoup(response.text, "html.parser")

                blocks = soup.find_all(["p", "div", "span"], text=True)
                raw_texts = clean_text_blocks(
                    [b.get_text(strip=True) for b in blocks]
                )
        except Exception as e:
            st.error(f"❌ Failed to fetch page: {e}")

# --------------------------------------------------
# 5. Text Type Classification
# --------------------------------------------------
classified_rows = []

if raw_texts:
    with st.spinner("🧠 Classifying text types..."):
        for text in raw_texts[:80]:  # safety limit
            text_type = classify_text_type(text)

            if text_type in ["customer complaint", "customer praise"]:
                classified_rows.append({
                    "text": text,
                    "type": text_type
                })

# --------------------------------------------------
# 6. Sentiment Analysis (ONLY on feedback)
# --------------------------------------------------
if classified_rows:
    with st.spinner("🤖 Running sentiment analysis..."):
        texts = [r["text"] for r in classified_rows]
        results = sentiment_pipe(texts, batch_size=16, truncation=True)

        df = pd.DataFrame({
            "text": texts,
            "category": [r["type"] for r in classified_rows],
            "sentiment": [
                normalize_sentiment(LABEL_MAP[r["label"]], r["score"])
                for r in results
            ],
            "confidence": [round(r["score"], 3) for r in results]
        })

    st.divider()

    # --------------------------------------------------
    # 7. Metrics
    # --------------------------------------------------
    pos = (df.sentiment == "POSITIVE").sum()
    neu = (df.sentiment == "NEUTRAL").sum()
    neg = (df.sentiment == "NEGATIVE").sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Positive", pos)
    m2.metric("Neutral", neu)
    m3.metric("Negative", neg)

    if neg > pos:
        st.error("❌ Customer dissatisfaction detected")
    elif pos > neg:
        st.success("✅ Overall customer response is positive")
    else:
        st.warning("⚠️ Mixed customer feedback")

    # --------------------------------------------------
    # 8. Sentiment Distribution
    # --------------------------------------------------
    st.divider()
    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df.sentiment.value_counts())

    # --------------------------------------------------
    # 9. Company Improvement Recommendations
    # --------------------------------------------------
    st.divider()
    st.subheader("🛠️ Company Improvement Recommendations")

    negative_text = " ".join(
        df[df.sentiment == "NEGATIVE"]["text"]
    ).lower()

    improvement_rules = {
        "Delivery & Logistics": (
            ["delivery", "late", "delay", "shipping", "courier"],
            "Improve delivery timelines and logistics coordination."
        ),
        "Product Quality": (
            ["quality", "broken", "damaged", "defective"],
            "Enhance product quality checks and durability."
        ),
        "Pricing": (
            ["expensive", "price", "overpriced"],
            "Revisit pricing strategy or offer better value."
        ),
        "Customer Support": (
            ["support", "service", "response", "help"],
            "Improve customer support responsiveness."
        ),
        "Returns & Refunds": (
            ["refund", "return", "replacement", "warranty"],
            "Simplify return and refund processes."
        )
    }

    found = False
    for area, (keywords, action) in improvement_rules.items():
        if any(k in negative_text for k in keywords):
            st.write(f"🔧 **{area}:** {action}")
            found = True

    if not found and neg > 0:
        st.info(
            "Negative feedback exists, but issues are spread across minor areas "
            "without a dominant problem."
        )

    # --------------------------------------------------
    # 10. Review Table
    # --------------------------------------------------
    with st.expander("🔍 View Analyzed Feedback"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("ℹ️ No valid customer feedback detected (product details and info ignored).")



        #python -m streamlit run app.py