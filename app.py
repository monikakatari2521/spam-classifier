import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #7C5CFF, #00D4FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title { color: #9AA4B2; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib"), joblib.load("vectorizer.joblib")


model, vec = load_model()
words = vec.get_feature_names_out()

EXAMPLES = {
    "🎁 Prize": "Congratulations! You won a free iPhone. Click now to claim your prize!",
    "💬 Friend": "Hey, are we meeting tomorrow at 5?",
    "🏦 Bank": "URGENT! Your account is blocked. Call 0800 123 456 now to claim your cash prize.",
    "📚 College": "Can you send me the notes for digital logic?",
}

if "msg" not in st.session_state:
    st.session_state.msg = ""
if "history" not in st.session_state:
    st.session_state.history = []


def set_example(text):
    st.session_state.msg = text


def analyze(text):
    X = vec.transform([text])
    spam_p = model.predict_proba(X)[0][1]
    diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    idx = X.nonzero()[1]
    contrib = [(words[i], X[0, i] * diff[i]) for i in idx]
    contrib = [c for c in contrib if c[1] > 0]
    contrib.sort(key=lambda c: c[1], reverse=True)
    return spam_p, contrib[:5]


st.markdown('<p class="main-title">🛡️ SMS Spam Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Machine learning model using TF-IDF and Naive Bayes</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔍 Check a message", "📂 Check many messages", "ℹ️ How it works"])

# ---------------- Tab 1: single message ----------------
with tab1:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(text,), use_container_width=True)

    st.text_area("Your message", key="msg", height=120, placeholder="Type or paste a message here...")

    if st.button("🔍 Check message", type="primary", use_container_width=True):
        text = st.session_state.msg
        if text.strip() == "":
            st.warning("Please type a message first.")
        else:
            spam_p, top = analyze(text)
            is_spam = spam_p >= 0.5

            if is_spam:
                st.error("🚨 This looks like SPAM")
            else:
                st.success("✅ This looks like a normal message")

            st.progress(float(spam_p), text=f"Spam probability: {spam_p * 100:.1f}%")

            if top:
                st.markdown("**Words that pushed it towards spam:**")
                st.write("  ".join(f"`{w}`" for w, _ in top))

            st.session_state.history.insert(
                0,
                {
                    "Message": text[:60] + ("..." if len(text) > 60 else ""),
                    "Verdict": "SPAM" if is_spam else "Normal",
                    "Spam %": round(spam_p * 100, 1),
                },
            )
            st.session_state.history = st.session_state.history[:10]

    if st.session_state.history:
        st.markdown("#### 🕘 Recent checks")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------- Tab 2: many messages ----------------
with tab2:
    st.write(
        "Upload a **.txt** file (one message per line) or a **.csv** file "
        "(messages in a column named `text`, or in the first column)."
    )
    file = st.file_uploader("Upload your messages", type=["txt", "csv"])

    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="latin-1")
            col = "text" if "text" in df.columns else df.columns[0]
            msgs = df[col].astype(str).tolist()
        else:
            raw = file.read().decode("latin-1")
            msgs = [line.strip() for line in raw.splitlines() if line.strip()]

        if msgs:
            probs = model.predict_proba(vec.transform(msgs))[:, 1]
            out = pd.DataFrame(
                {
                    "message": msgs,
                    "spam %": (probs * 100).round(1),
                    "verdict": np.where(probs >= 0.5, "SPAM", "Normal"),
                }
            )
            n_spam = int((probs >= 0.5).sum())
            st.metric("Spam messages found", f"{n_spam} of {len(msgs)}")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download results", out.to_csv(index=False), "spam_results.csv")
        else:
            st.warning("No messages found in the file.")

# ---------------- Tab 3: how it works ----------------
with tab3:
    st.markdown(
        """
        ### How this app works
        1. **Dataset:** SMS Spam Collection, about 5,500 labelled messages.
        2. **TF-IDF:** converts each message into numbers. Words that are rare overall
           but frequent in a message (like *free* or *winner*) get high scores.
        3. **Naive Bayes:** learns which words appear more often in spam than in normal messages.
        4. **Prediction:** the model gives a probability that the message is spam.
        5. **Trigger words:** shown by comparing how strongly each word points to spam vs normal.

        ### Model results (test set)
        | Metric | Value |
        |---|---|
        | Accuracy | 97.8% |
        | Spam precision | 1.00 |
        | Spam recall | 0.84 |

        **Note:** this model was trained on older SMS data, so modern spam may sometimes slip through.
        """
    )
