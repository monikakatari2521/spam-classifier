import streamlit as st
import joblib

model = joblib.load("spam_model.joblib")
vec = joblib.load("vectorizer.joblib")

st.title("SMS Spam Detector")
st.write("Type a message and click Check.")

message = st.text_area("Your message")

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please type a message first.")
    else:
        result = model.predict(vec.transform([message]))[0]
        if result == 1:
            st.error("This looks like SPAM")
        else:
            st.success("This looks like a normal message")
