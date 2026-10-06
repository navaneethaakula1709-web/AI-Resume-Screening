# AI Resume Screening System

## 📌 Project Overview

The AI Resume Screening System is an AI-assisted application designed to help recruiters screen multiple resumes against a given Job Description.

The system extracts text from PDF resumes, identifies relevant skills, education and experience information, and calculates a resume-to-job match score using Natural Language Processing (NLP) techniques.

## 🎯 Objectives

- Extract text from PDF resumes
- Identify candidate skills
- Extract education information
- Extract work experience information
- Identify required skills from the Job Description
- Compare resumes with the Job Description
- Calculate a match score
- Rank candidates based on their match score
- Provide a simple recruiter-friendly interface

## 🚀 Features

- Job Description input
- Multiple PDF resume upload
- Resume text extraction
- Skill extraction
- Education extraction
- Experience extraction
- TF-IDF based text representation
- Cosine Similarity based matching
- Candidate ranking
- Match score display
- High, Moderate and Low Match status
- Streamlit web interface

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Natural Language Processing (NLP)

## ⚙️ How the System Works

1. Recruiter enters the Job Description.
2. Recruiter uploads one or more PDF resumes.
3. The system extracts text from each resume.
4. Candidate skills, education and experience are identified.
5. Required skills are identified from the Job Description.
6. TF-IDF converts the Job Description and resume text into numerical vectors.
7. Cosine Similarity calculates the similarity between them.
8. A match score is generated for each candidate.
9. Candidates are ranked based on their match score.
10. Results are displayed through the Streamlit interface.

## 📊 Matching Algorithm

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to represent text and **Cosine Similarity** to measure the similarity between the Job Description and each resume.

The similarity score is converted into a percentage and displayed as the candidate's Match Score.

## ▶️ How to Run

### 1. Install required packages

```bash
pip install -r requirements.txt