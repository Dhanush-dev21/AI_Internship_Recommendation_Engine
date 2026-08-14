# 🎯 AI Internship Recommendation Engine

An AI-powered recommendation system that provides personalized **internship opportunities, project ideas, and learning resources** based on a student's domain, experience level, skills, and interests.

## 📌 Project Information

* **Task ID:** AI-SS-002
* **Student Code:** DAS005800
* **Project:** AI Internship Recommendation Engine
* **Technology:** Python, Flask, Pandas, NumPy, Scikit-learn
* **Recommendation Types:** Internship, Project, Learning Resource

## 📖 Project Description

The AI Internship Recommendation Engine is a web-based recommendation system designed to help students discover suitable internships, projects, and learning resources.

The system accepts a student's:

* Name
* Domain
* Experience level
* Skills
* Interests

It analyzes this profile against available internship, project, and learning-resource datasets and generates the **Top 5 recommendations** with match scores.

The internship recommendation system combines:

1. **Content-Based Filtering**
2. **Collaborative Filtering**
3. **Hybrid Recommendation**

Project and learning-resource recommendations use profile-based content matching.

## ✨ Features

* 🎓 Student profile input
* 💼 Top 5 internship recommendations
* 💻 Top 5 project recommendations
* 📚 Top 5 learning-resource recommendations
* 🧠 Content-based filtering
* 👥 Collaborative filtering
* 🔀 Hybrid recommendation
* 📊 TF-IDF text similarity
* 📐 Cosine similarity
* 🎯 Domain matching
* 📈 Experience-level matching
* ⭐ Match scores
* 📉 Recommendation evaluation using MAE and RMSE
* 🌐 Flask web interface
* 📱 Responsive web design
* 📄 CSV-based datasets
* 🔄 Try another profile functionality

## 🧠 Recommendation Approach

### 1. Content-Based Filtering

Content-based filtering compares the student's profile with the characteristics of internships, projects, and learning resources.

The student's profile contains:

* Domain
* Experience level
* Skills
* Interests

These are compared with:

* Internship domain
* Required internship skills
* Experience requirements
* Project domain
* Project required skills
* Project difficulty
* Resource domain
* Resource topics
* Resource level

The system calculates similarity scores and ranks the recommendations.

### 2. TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text information into numerical vectors.

It is used to represent the student's profile and recommendation-item information so that their similarity can be calculated.

### 3. Cosine Similarity

Cosine similarity measures the similarity between the student's profile vector and recommendation-item vectors.

A higher similarity indicates that the recommendation is more relevant to the student's profile.

### 4. Collaborative Filtering

Collaborative filtering uses the ratings stored in `ratings.csv`.

The system:

1. Creates a student-internship rating matrix.
2. Calculates similarity between students.
3. Finds students with similar rating patterns.
4. Uses their ratings to estimate recommendation scores.

This allows the system to use the preferences of similar students.

### 5. Hybrid Recommendation

The internship recommendation combines content-based and collaborative filtering.

The current hybrid formula uses:

```text
Hybrid Score = 60% Content Score + 40% Collaborative Score
```

This combines:

* The student's own skills, interests, domain, and experience
* The preferences of similar students

The internships are then sorted by their hybrid score and the **Top 5** are displayed.

## 📊 Recommendation Evaluation

The recommendation engine includes evaluation using:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual ratings and predicted ratings.

Current evaluation result:

```text
MAE: 0.6936
```

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

Current evaluation result:

```text
RMSE: 1.0444
```

These metrics are calculated using the available student-internship ratings in `ratings.csv`.

## 📂 Datasets

### `students.csv`

Contains student information used to identify students for collaborative filtering.

### `internships.csv`

Contains internship information including:

* Internship ID
* Title
* Company
* Domain
* Required skills
* Experience level
* Duration
* Location

### `ratings.csv`

Contains student ratings for internships.

Example structure:

```text
student_id,internship_id,rating
S001,I001,5
S001,I002,4
S002,I003,5
```

### `projects.csv`

Contains project recommendations including:

* Project ID
* Title
* Domain
* Required skills
* Difficulty

### `resources.csv`

Contains learning resources including:

* Resource ID
* Title
* Domain
* Topics
* Level
* Type

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **HTML**
* **CSS**
* **CSV**

## 📁 Project Structure

```text
AI_Internship_Recommendation_Engine/
│
├── app.py
├── recommender.py
├── main.py
│
├── students.csv
├── internships.csv
├── ratings.csv
├── projects.csv
├── resources.csv
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   └── results.html
│
└── static/
    └── style.css
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dhanush-dev21/AI_Internship_Recommendation_Engine.git
```

### 2. Open the project directory

```bash
cd AI_Internship_Recommendation_Engine
```

### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will run at:

```text
http://127.0.0.1:5000
```

Open this address in a web browser.

## 🧪 Example Input

```text
Name: Frank
Domain: Data
Experience: intermediate
Skills: Python, Pandas, Numpy, SQL
Interests: Data Science, Analytics
```

## 📌 Example Results

The system generates three recommendation sections.

### 💼 Internship Recommendations

Example:

```text
1. Data Science Intern
Company: DataWorks
Domain: Data
Experience: intermediate
Duration: 3 months
Location: Remote
Match Score: 88.68%
```

### 💻 Project Recommendations

Example:

```text
1. Data Analytics Dashboard
Domain: Data
Required Skills: python,pandas,sql
Difficulty: intermediate
Match Score: 85.74%
```

### 📚 Learning Resources

Example:

```text
1. Pandas and NumPy for Data Science
Domain: Data
Topics: python,pandas,numpy,data science
Level: beginner
Match Score: 89.47%
```

## 🌐 GitHub Repository

[AI Internship Recommendation Engine](https://github.com/Dhanush-dev21/AI_Internship_Recommendation_Engine)

## 🎥 YouTube Demonstration

**YouTube Video:** To be added after the project demonstration video is uploaded.

```text
YouTube Link: [Add your active YouTube URL here]
```

## 📝 Project Task

* **Task ID:** AI-SS-002
* **Student Code:** DAS005800
* **Task:** AI Internship Recommendation Engine

**Task requirements include:**

* Internship recommendations
* Collaborative filtering
* Content-based filtering
* Hybrid recommendation
* Data preprocessing
* Evaluation metrics
* Clean and documented code
* GitHub repository
* Project report
* Screenshots
* Demonstration video
* Blog submission

## 📈 Git Commit History

The project was developed incrementally using Git.

Major commits include:

```text
b31758e Add recommendation evaluation metrics
be84f78 Finalize recommendation engine logic
7125fd8 Add hybrid project and resource recommendations
f165f19 Add project and learning resource recommendations
493b1e0 Initial commit - AI Internship Recommendation Engine
```

This commit history demonstrates the development progress of the project.

## 🚀 Future Improvements

Possible future improvements include:

* Larger internship and student datasets
* More advanced recommendation algorithms
* Improved collaborative filtering
* Neural-network-based recommendation models
* User authentication
* Database integration
* Recommendation history
* Feedback and rating system
* Real-time internship data
* Advanced evaluation metrics
* Deployment to a cloud platform

## 👨‍💻 Author

**Dhanush-dev21**

Computer Science Student

## 📄 License

This project was developed as part of an internship task for educational and project-development purposes.
