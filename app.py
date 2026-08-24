from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="HotelEcho | Review intelligence",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "tripadvisor_hotel_reviews.csv"


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["Review", "Rating"])
    data["Rating"] = data["Rating"].astype(int)

    reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(
        data["Review"], data["Rating"], test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(reviews_train)
    test_vectors = vectorizer.transform(reviews_test)
    model = MultinomialNB()
    model.fit(train_vectors, ratings_train)
    accuracy = accuracy_score(ratings_test, model.predict(test_vectors))
    return data, vectorizer, model, accuracy


def rating_label(rating):
    labels = {1: "Very poor", 2: "Needs work", 3: "Mixed stay", 4: "Good stay", 5: "Exceptional"}
    return labels.get(int(rating), "Predicted rating")


def render_stars(rating):
    return "★" * int(rating) + "☆" * (5 - int(rating))


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17221f; --muted:#687570; --cream:#f7f5ef; --paper:#fffdf8; --mint:#c7e5d2; --coral:#ee745d; --line:#dce2dc; }
    .stApp { background:var(--cream); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#1b2b27; border-right:0; }
    [data-testid="stSidebar"] * { color:#f5f2e9 !important; }
    [data-testid="stSidebar"] .stCaption { color:#b9c7bf !important; }
    .block-container { max-width:1320px; padding:3.5rem 4rem 4rem; }
    h1,h2,h3,p,div,span,label { font-family:'Manrope',sans-serif; }
    h1 { font-size:clamp(2.5rem, 5vw, 5.5rem) !important; line-height:.98 !important; letter-spacing:-.06em !important; font-weight:800 !important; }
    h2 { letter-spacing:-.04em; }
    .eyebrow { color:var(--coral); font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; margin-bottom:1rem; }
    .hero-copy { max-width:720px; color:var(--muted); font-size:1.05rem; line-height:1.7; margin:1.4rem 0 2.4rem; }
    .hero-mark { display:inline-flex; background:var(--mint); color:var(--ink); padding:.35rem .65rem; border-radius:3px; transform:rotate(-2deg); }
    .panel { background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:1.5rem; }
    .panel-label { color:var(--muted); font-family:'DM Mono',monospace; font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; }
    .score { font-size:5rem; font-weight:800; line-height:1; letter-spacing:-.08em; color:var(--ink); }
    .score-stars { color:var(--coral); font-size:1.3rem; letter-spacing:.12em; }
    .insight { border-left:3px solid var(--coral); padding-left:1rem; color:var(--muted); line-height:1.6; }
    .stButton > button { background:var(--ink); color:white; border:0; border-radius:4px; min-height:3rem; font-weight:700; }
    .stButton > button:hover { background:var(--coral); color:white; }
    textarea { background:var(--paper) !important; border-color:var(--line) !important; }
    [data-testid="stMetric"] { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:1rem; }
    [data-testid="stMetricLabel"] { font-family:'DM Mono',monospace; text-transform:uppercase; font-size:.66rem; }
    [data-testid="stMetricValue"] { font-weight:800; }
    .footer { color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); padding-top:1.2rem; margin-top:3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.spinner("Training your review model..."):
    data, vectorizer, model, accuracy = train_model()

with st.sidebar:
    st.markdown("## HotelEcho")
    st.caption("Review intelligence, made legible.")
    st.divider()
    st.markdown("**Model status**")
    st.success("Online")
    st.caption(f"Naive Bayes  /  {len(vectorizer.vocabulary_):,} learned terms")
    st.divider()
    st.markdown("**Navigate**")
    view = st.radio("Navigate", ["Review analyzer", "Dataset signals"], label_visibility="collapsed")
    st.divider()
    st.caption("Built from Tripadvisor hotel reviews")


if view == "Review analyzer":
    st.markdown('<div class="eyebrow">Hotel review intelligence / 01</div>', unsafe_allow_html=True)
    st.title("Turn a stay into a signal.")
    st.markdown(
        '<p class="hero-copy">Paste a guest review and HotelEcho will estimate the rating it communicates. Explore the reasoning through confidence, not just a single number.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown('<div class="panel-label">Guest review</div>', unsafe_allow_html=True)
        examples = [
            "The room was spotless and the staff made us feel completely at home. Perfect location.",
            "Great location, but the room was noisy and the service felt indifferent.",
            "The bed was comfortable and the hotel was fine for one night.",
        ]
        selected = st.selectbox("Try an example", ["Write your own"] + examples, label_visibility="collapsed")
        default_text = "" if selected == "Write your own" else selected
        review = st.text_area(
            "Review text",
            value=default_text,
            height=230,
            placeholder="Tell us about the room, service, location, or the little things...",
            label_visibility="collapsed",
        )
        analyze = st.button("Analyze review  ->", use_container_width=True)
        if analyze and review.strip():
            review_vector = vectorizer.transform([review])
            predicted = int(model.predict(review_vector)[0])
            probabilities = model.predict_proba(review_vector)[0]
            confidence = float(max(probabilities))
            st.session_state["prediction"] = (predicted, confidence, probabilities)
        elif analyze:
            st.warning("Add a few words before analyzing.")

        st.markdown("<div class='insight'>The model reads the language of the review and compares it with patterns learned from thousands of real hotel experiences.</div>", unsafe_allow_html=True)

    with right:
        prediction = st.session_state.get("prediction")
        if prediction:
            predicted, confidence, probabilities = prediction
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-label">Predicted experience</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score">{predicted}<span style="font-size:1.4rem;letter-spacing:0"> / 5</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-stars">{render_stars(predicted)}</div>', unsafe_allow_html=True)
            st.subheader(rating_label(predicted))
            st.progress(confidence, text=f"Model confidence  {confidence:.0%}")
            st.markdown("**Rating profile**")
            chart = pd.DataFrame({"Probability": probabilities}, index=[f"{n} star" for n in model.classes_])
            st.bar_chart(chart, horizontal=True, height=180, color="#ee745d")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel" style="min-height:390px;display:flex;align-items:center;justify-content:center;text-align:center"><div><div style="font-size:3rem;margin-bottom:1rem">✦</div><h3>Your result will appear here</h3><p style="color:#687570">A clear rating estimate, with the nuance behind it.</p></div></div>', unsafe_allow_html=True)

    st.markdown("### Model at a glance")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Reviews learned", f"{len(data):,}")
    metric_two.metric("Validation accuracy", f"{accuracy:.1%}")
    metric_three.metric("Rating range", "1 - 5 stars")

else:
    st.markdown('<div class="eyebrow">Hotel review intelligence / 02</div>', unsafe_allow_html=True)
    st.title("The shape of a stay.")
    st.markdown('<p class="hero-copy">A quick read of the review collection behind the model. Use the distribution to understand what “typical” looks like before you interpret an individual voice.</p>', unsafe_allow_html=True)
    distribution = data["Rating"].value_counts().sort_index().rename("Reviews")
    col_a, col_b = st.columns([1.2, .8], gap="large")
    with col_a:
        st.markdown('<div class="panel-label">Rating distribution</div>', unsafe_allow_html=True)
        st.bar_chart(distribution, height=350, color="#c7e5d2")
    with col_b:
        st.markdown('<div class="panel-label">Collection profile</div>', unsafe_allow_html=True)
        st.metric("Average rating", f"{data.Rating.mean():.2f} / 5")
        st.metric("Most common rating", f"{int(data.Rating.mode().iloc[0])} stars")
        st.metric("Average review length", f"{int(data.Review.str.split().str.len().mean())} words")
        st.markdown('<div class="insight">Ratings are human summaries. The review text carries the context that makes them useful.</div>', unsafe_allow_html=True)
    st.markdown("### A few voices")
    samples = data.sample(3, random_state=12)[["Review", "Rating"]].reset_index(drop=True)
    for _, sample in samples.iterrows():
        st.markdown(f'<div class="panel" style="margin-bottom:.7rem"><span class="score-stars">{render_stars(sample.Rating)}</span><p style="margin:.6rem 0 0;color:#687570;line-height:1.6">{sample.Review[:420]}...</p></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">HotelEcho uses a Multinomial Naive Bayes classifier with CountVectorizer. Predictions are estimates, not replacements for reading the full story.</div>', unsafe_allow_html=True)