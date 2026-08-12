# 🎯 AI Internship Recommendation Engine

An AI-powered web application that recommends suitable internships to students based on their skills, interests, domain, and experience level.

## 📌 Project Description

The AI Internship Recommendation Engine uses content-based recommendation techniques to compare a student's profile with available internship opportunities.

The system analyzes:

- Student domain
- Experience level
- Skills
- Interests
- Internship required skills
- Internship domain
- Internship experience requirements

It then calculates a match score and displays the top 5 recommended internships.

## ✨ Features

- Student profile input through a web interface
- Personalized internship recommendations
- TF-IDF text-based similarity
- Cosine similarity for matching
- Domain matching
- Experience-level matching
- Top 5 internship recommendations
- Match percentage for each recommendation
- Internship company, duration, and location
- Responsive web interface
- Multiple sample student profiles for testing

## 🧠 Recommendation Method

The system uses a content-based recommendation approach.

### TF-IDF

TF-IDF converts the student's profile and internship information into numerical vectors.

### Cosine Similarity

Cosine similarity measures how closely the student's profile matches each internship.

The system also applies additional scoring bonuses when the student's:

- Domain matches the internship domain
- Experience level matches the internship requirement

The internships are then sorted by their final match score.

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML
- CSS
- CSV datasets

## 📂 Project Structure

```text
AI_Internship_Recommendation_Engine/
│
├── app.py
├── recommender.py
├── students.csv
├── internships.csv
├── ratings.csv
├── projects.csv
├── resources.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── results.html
│
└── static/
    └── style.css