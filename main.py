import os
import spacy
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import fitz  # PyMuPDF for PDF processing
from docx import Document
import re

# Ensure the spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load Sentence Transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face's NER model (Replace with JobBERT if available)
ner_model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

# Expanded Predefined Skills List
PREDEFINED_SKILLS = set([
    "Python", "Java", "C++", "JavaScript", "SQL", "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "Data Science", "NLP", "TensorFlow", "PyTorch", "Keras",
    "Flask", "Django", "FastAPI", "React", "Angular", "Vue.js", "Node.js",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Git", "DevOps",
    "Cybersecurity", "Blockchain", "Big Data", "Linux", "Unix", "Embedded Systems",
    "Agile", "Scrum", "JIRA", "Power BI", "Tableau", "Software Testing",
    "Android Development", "iOS Development", "React Native", "Flutter",
    "Natural Language Processing", "Computer Vision", "MLOps", "ETL", "Data Engineering"
])

# Ex
