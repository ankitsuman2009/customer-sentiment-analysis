# ============================================================
# CUSTOMER SENTIMENT ANALYSIS SYSTEM
# NLP + RoBERTa + Streamlit + Pandas + Plotly
# ============================================================

import re
import warnings

import pandas as pd
import streamlit as st
import plotly.express as px

from transformers import pipeline

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.markdown(
    '<div class="main-title">Customer Sentiment Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent NLP & Machine Learning Based Customer Feedback Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Project Settings")

st.sidebar.info(
    """
    This application uses Natural Language Processing
    and a Transformer-based RoBERTa model to analyze
    customer feedback.
    """
)

st.sidebar.markdown("### Technologies")

st.sidebar.write("🐍 Python")
st.sidebar.write("🤗 Hugging Face Transformers")
st.sidebar.write("🧠 RoBERTa")
st.sidebar.write("🌐 Streamlit")
st.sidebar.write("📊 Pandas")
st.sidebar.write("📈 Plotly")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_sentiment_model():

    """
    Load pretrained 3-class RoBERTa sentiment model.

    Classes:
        Negative
        Neutral
        Positive
    """

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )

    return classifier


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading sentiment model..."):

    try:

        sentiment_model = load_sentiment_model()

        st.sidebar.success(
            "3-class RoBERTa model loaded successfully!"
        )

    except Exception as error:

        st.error(
            "Model load nahi ho paya."
        )

        st.code(str(error))

        st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    """
    Clean raw customer feedback.

    Operations:
        1. Convert to string
        2. Remove URLs
        3. Remove HTML tags
        4. Remove mentions
        5. Remove extra spaces
    """

    if text is None:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    """
    Predict sentiment using 3-class RoBERTa model.

    Output:
        Positive
        Negative
        Neutral
    """

    cleaned = clean_text(text)

    if not cleaned:

        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "raw_label": "EMPTY",
            "cleaned_text": ""
        }

    try:

        result = sentiment_model(
            cleaned,
            truncation=True,
            max_length=512
        )[0]

        raw_label = str(
            result["label"]
        ).upper()

        confidence = float(
            result["score"]
        )

        # ====================================================
        # LABEL MAPPING
        # ====================================================

        if "NEGATIVE" in raw_label:

            sentiment = "Negative"

        elif "NEUTRAL" in raw_label:

            sentiment = "Neutral"

        elif "POSITIVE" in raw_label:

            sentiment = "Positive"

        elif raw_label == "LABEL_0":

            sentiment = "Negative"

        elif raw_label == "LABEL_1":

            sentiment = "Neutral"

        elif raw_label == "LABEL_2":

            sentiment = "Positive"

        else:

            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_label": raw_label,
            "cleaned_text": cleaned
        }

    except Exception as error:

        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "raw_label": "ERROR",
            "cleaned_text": cleaned,
            "error": str(error)
        }


# ============================================================
# SENTIMENT EMOJI
# ============================================================

def sentiment_emoji(sentiment):

    if sentiment == "Positive":

        return "😊"

    elif sentiment == "Negative":

        return "😞"

    else:

        return "😐"


# ============================================================
# SENTIMENT DESCRIPTION
# ============================================================

def sentiment_description(sentiment):

    if sentiment == "Positive":

        return "Customer feedback is positive."

    elif sentiment == "Negative":

        return "Customer feedback is negative."

    else:

        return "Customer feedback is neutral."


# ============================================================
# SINGLE TEXT ANALYSIS (MAIN UI)
# ============================================================

st.header(
    "🔍 Single Customer Review Analysis"
)

st.write(
    "Enter a customer review and press Enter "
    "or click Analyze Sentiment."
)

# ============================================================
# FORM
# ============================================================

with st.form(
    "single_sentiment_form"
):

    text = st.text_input(
        "Customer Feedback",
        placeholder=(
            "Example: The product quality is excellent "
            "and delivery was very fast."
        )
    )

    analyze_button = st.form_submit_button(
        "🔎 Analyze Sentiment",
        type="primary"
    )

# ============================================================
# ANALYSIS EXECUTION
# ============================================================

if analyze_button:

    if not text.strip():

        st.warning(
            "Please enter customer feedback first."
        )

    else:

        with st.spinner(
            "Analyzing customer feedback..."
        ):

            result = predict_sentiment(text)

        sentiment = result["sentiment"]

        confidence = result["confidence"]

        st.divider()

        # ====================================================
        # RESULT COLUMNS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Sentiment",
                sentiment_emoji(sentiment)
                + " "
                + sentiment
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Model",
                "RoBERTa"
            )

        # ====================================================
        # RESULT MESSAGE
        # ====================================================

        st.subheader(
            "📋 Analysis Result"
        )

        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment_description(sentiment)}"
            )

        elif sentiment == "Negative":

            st.error(
                f"😞 {sentiment_description(sentiment)}"
            )

        else:

            st.info(
                f"😐 {sentiment_description(sentiment)}"
            )

        # ====================================================
        # CLEANED TEXT
        # ====================================================

        st.subheader(
            "🧹 Cleaned Text"
        )

        st.write(
            result["cleaned_text"]
        )

        # ====================================================
        # RAW MODEL LABEL
        # ====================================================

        st.subheader(
            "🤖 Model Prediction"
        )

        st.write(
            f"Raw Label: `{result['raw_label']}`"
        )

        # ====================================================
        # CONFIDENCE CHART
        # ====================================================

        st.subheader(
            "📊 Model Confidence"
        )

        confidence_data = pd.DataFrame(
            {
                "Metric": ["Confidence"],
                "Value": [confidence]
            }
        )

        fig = px.bar(
            confidence_data,
            x="Metric",
            y="Value",
            range_y=[0, 1],
            title="Prediction Confidence"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )    