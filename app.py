import streamlit as st
import joblib

st.set_page_config(page_title="Spam Classifier", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

st.title("🛡️ Spam Classifier")
st.markdown("### Naïve Bayes — Bayesian Probabilistic Analysis")
st.divider()

user_input = st.text_area("Enter a message:", height=150, placeholder="Type or paste a message here...")

if st.button("Classify", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        st.divider()
        st.markdown("### Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classification", "🚨 SPAM" if prediction == 1 else "✅ HAM")
        with col2:
            st.metric("Spam Probability", f"{probs[1]*100:.1f}%")
        with col3:
            st.metric("Ham Probability", f"{probs[0]*100:.1f}%")
