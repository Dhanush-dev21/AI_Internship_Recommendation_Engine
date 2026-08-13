from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# ============================================================
# LOAD DATA
# ============================================================

students = pd.read_csv("students.csv")
internships = pd.read_csv("internships.csv")
ratings = pd.read_csv("ratings.csv")
projects = pd.read_csv("projects.csv")
resources = pd.read_csv("resources.csv")


# ============================================================
# CONTENT-BASED INTERNSHIP RECOMMENDATION
# ============================================================

def content_scores(domain, experience, skills, interests):

    user_profile = (
        str(domain).lower() + " "
        + str(experience).lower() + " "
        + str(skills).lower() + " "
        + str(interests).lower()
    )

    internship_profiles = []

    for _, row in internships.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["experience_level"]).lower() + " "
            + str(row["required_skills"]).lower()
        )

        internship_profiles.append(profile)

    documents = [user_profile] + internship_profiles

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    scores = {}

    for index, row in internships.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if str(row["domain"]).lower() == str(domain).lower():
            score += 10

        # Experience bonus
        if (
            str(row["experience_level"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        scores[row["internship_id"]] = score

    return scores


# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

def collaborative_scores(student_id):

    rating_matrix = ratings.pivot_table(
        index="student_id",
        columns="internship_id",
        values="rating",
        fill_value=0
    )

    # New student / unknown student
    if student_id not in rating_matrix.index:

        return {
            internship_id: 0
            for internship_id in internships["internship_id"]
        }

    student_similarity = cosine_similarity(rating_matrix)

    similarity_df = pd.DataFrame(
        student_similarity,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

    similar_students = similarity_df.loc[student_id]

    similar_students = similar_students.drop(
        student_id,
        errors="ignore"
    )

    scores = {}

    for internship_id in internships["internship_id"]:

        weighted_sum = 0
        similarity_sum = 0

        for other_student, similarity in similar_students.items():

            rating = rating_matrix.loc[
                other_student,
                internship_id
            ]

            if rating > 0:

                weighted_sum += similarity * rating
                similarity_sum += similarity

        if similarity_sum > 0:
            predicted_rating = (
                weighted_sum / similarity_sum
            )
        else:
            predicted_rating = 0

        scores[internship_id] = predicted_rating

    return scores


# ============================================================
# PROJECT RECOMMENDATIONS
# ============================================================

def project_recommendations(domain, experience, skills, interests):

    user_text = (
        str(domain).lower() + " "
        + str(experience).lower() + " "
        + str(skills).lower() + " "
        + str(interests).lower()
    )

    project_profiles = []

    for _, row in projects.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["difficulty"]).lower() + " "
            + str(row["required_skills"]).lower()
        )

        project_profiles.append(profile)

    documents = [user_text] + project_profiles

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    results = []

    for index, row in projects.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if str(row["domain"]).lower() == str(domain).lower():
            score += 10

        # Experience / difficulty bonus
        if (
            str(row["difficulty"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        results.append({
            "title": row["title"],
            "domain": row["domain"],
            "required_skills": row["required_skills"],
            "difficulty": row["difficulty"],
            "score": score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# LEARNING RESOURCE RECOMMENDATIONS
# ============================================================

def resource_recommendations(domain, experience, skills, interests):

    user_text = (
        str(domain).lower() + " "
        + str(experience).lower() + " "
        + str(skills).lower() + " "
        + str(interests).lower()
    )

    resource_profiles = []

    for _, row in resources.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["level"]).lower() + " "
            + str(row["topics"]).lower()
        )

        resource_profiles.append(profile)

    documents = [user_text] + resource_profiles

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    results = []

    for index, row in resources.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if str(row["domain"]).lower() == str(domain).lower():
            score += 10

        # Level bonus
        if (
            str(row["level"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        results.append({
            "title": row["title"],
            "domain": row["domain"],
            "topics": row["topics"],
            "level": row["level"],
            "type": row["type"],
            "score": score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# HYBRID INTERNSHIP RECOMMENDATION
# 60% CONTENT + 40% COLLABORATIVE
# ============================================================

def get_recommendations(
    name,
    domain,
    experience,
    skills,
    interests
):

    # --------------------------------------------------------
    # Find student
    # --------------------------------------------------------

    student = students[
        students["name"].str.lower()
        == name.lower()
    ]

    if len(student) > 0:
        student_id = student.iloc[0]["student_id"]
    else:
        student_id = None


    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content = content_scores(
        domain,
        experience,
        skills,
        interests
    )


    # --------------------------------------------------------
    # Collaborative scores
    # --------------------------------------------------------

    if student_id is not None:

        collaborative = collaborative_scores(
            student_id
        )

    else:

        collaborative = {
            internship_id: 0
            for internship_id in internships["internship_id"]
        }


    # --------------------------------------------------------
    # Normalize collaborative scores
    # --------------------------------------------------------

    max_collaborative = max(
        collaborative.values()
    ) if collaborative else 0

    normalized_collaborative = {}

    for internship_id, score in collaborative.items():

        if max_collaborative > 0:

            normalized_score = (
                score / max_collaborative
            ) * 100

        else:

            normalized_score = 0

        normalized_collaborative[
            internship_id
        ] = normalized_score


    # --------------------------------------------------------
    # Hybrid score
    # 60% Content + 40% Collaborative
    # --------------------------------------------------------

    results = []

    for _, row in internships.iterrows():

        internship_id = row["internship_id"]

        content_score = content.get(
            internship_id,
            0
        )

        collaborative_score = normalized_collaborative.get(
            internship_id,
            0
        )

        hybrid_score = (
            0.60 * content_score
            + 0.40 * collaborative_score
        )

        results.append({

            "title": row["title"],

            "company": row["company"],

            "domain": row["domain"],

            "experience_level":
                row["experience_level"],

            "duration": row["duration"],

            "location": row["location"],

            "score": hybrid_score

        })


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    # Get form data

    name = request.form.get(
        "name",
        ""
    ).strip()

    domain = request.form.get(
        "domain",
        ""
    ).strip()

    experience = request.form.get(
        "experience",
        ""
    ).strip()

    skills = request.form.get(
        "skills",
        ""
    ).strip()

    interests = request.form.get(
        "interests",
        ""
    ).strip()


    # --------------------------------------------------------
    # Internship recommendations
    # --------------------------------------------------------

    recommendations = get_recommendations(
        name,
        domain,
        experience,
        skills,
        interests
    )


    # --------------------------------------------------------
    # Project recommendations
    # --------------------------------------------------------

    project_results = project_recommendations(
        domain,
        experience,
        skills,
        interests
    )


    # --------------------------------------------------------
    # Learning resource recommendations
    # --------------------------------------------------------

    resource_results = resource_recommendations(
        domain,
        experience,
        skills,
        interests
    )


    # --------------------------------------------------------
    # Send all results to results.html
    # --------------------------------------------------------

    return render_template(

        "results.html",

        name=name,

        domain=domain,

        experience=experience,

        skills=skills,

        interests=interests,

        recommendations=recommendations,

        project_results=project_results,

        resource_results=resource_results

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)