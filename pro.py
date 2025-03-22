#AI-Powerd Resume Screening & Ranking System
#Technology Used:-python,stremlit,NLP,pypdf2
#Python Program & Libary
#ML Libary
#Basic HTML & CSS For Designing
#NAME:-HALVADIA DARSHIL 

#First Download Required Libary For This Program.
import streamlit as st
import pandas as pd
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load External CSS for better UI
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
#RESUME RANKING SYSTEAM.!
# Sidebar Navigation With User Freindly
st.sidebar.markdown("""
    <div class='sidebar-container'>
        <h2>📌 Navigation</h2>
        <ul>
            <li>📂 Upload resumes (PDF)</li>
            <li>📝 Enter job description</li>
            <li>🎯 Click 'Rank Resumes'</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Title with Animation modern tourch
st.markdown("<div class='title-container'><h1>🚀 AI Resume Screening & Ranking</h1></div>", unsafe_allow_html=True)

# Job Description Input
job_description = st.text_area("✍️ Enter Your  Job Description Here")

# Resume Upload sectione
uploaded_files = st.file_uploader("📄 Upload Resumes (PDF Only)", accept_multiple_files=True, type=["pdf"])

# Extract Text from PDFs (PDF extract text from PDF)
def extract_text(file):
    pdf_reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

# Extract Keywords from Resume Bt Functione from Uploded Resume PDF.

def extract_keywords(text, top_n=5):
    words = text.lower().split()
    common_words = Counter(words).most_common(top_n)
    return [word[0] for word in common_words]

# Rank Resumes Based on Job Description
def rank_resumes(job_desc, resumes):
    if not job_desc or not resumes:
        return []
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

    scores = [score * 100 for score in similarities]
    ranked_resumes = list(zip(resumes, scores))
    
    return sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

# Rank Button
if st.button("🎯 Rank Resumes"):
    with st.spinner("🔍 Processing..."):
        time.sleep(2)  # Simulated Loading Effect
        resume_texts = [extract_text(f) for f in uploaded_files if f is not None]
        ranked = rank_resumes(job_description, resume_texts)

    # Display Results
    st.markdown("<div class='results-container'><h2>🏆 Ranked Resumes</h2></div>", unsafe_allow_html=True)
    ranked_data = []

    for i, (resume, score) in enumerate(ranked):
        color_class = "high-score" if score >= 60 else "medium-score" if score >= 35 else "low-score"
        top_keywords = extract_keywords(resume)

        # Display resume ranking with card effect in UI
        st.markdown(
            f"""
            <div class="resume-card {color_class}">
                <h3>📜 Resume {i+1}</h3>
                <p>🎯 Score: {score:.2f}/100</p>
                <p>🔑 Keywords: {', '.join(top_keywords)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        ranked_data.append({"Resume": f"Resume {i+1}", "Score": score, "Keywords": ", ".join(top_keywords)})

    # Download Option Your Resume Ranked to CSV.file 
    if ranked_data:
        df = pd.DataFrame(ranked_data)
        st.download_button("📥 Download Results", df.to_csv(index=False), "ranked_resumes.csv", "text/csv")
