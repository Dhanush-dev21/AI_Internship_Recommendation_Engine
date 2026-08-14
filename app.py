from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
# DATA PREPROCESSING
# ============================================================

def preprocess_dataframe(df):
    """
    Clean text columns while avoiding pandas 4 warnings.
    """

    df = df.copy()

    for column in df.select_dtypes(include=["str"]).columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


students = preprocess_dataframe(students)
internships = preprocess_dataframe(internships)
ratings = preprocess_dataframe(ratings)
projects = preprocess_dataframe(projects)
resources = preprocess_dataframe(resources)


# ============================================================
# CREATE USER PROFILE
# ============================================================

def create_user_profile(
    domain,
    experience,
    skills,
    interests
):
    """
    Create a combined text profile for the student.
    """

    return (
        str(domain).lower() + " "
        + str(experience).lower() + " "
        + str(skills).lower() + " "
        + str(interests).lower()
    )


# ============================================================
# CONTENT-BASED INTERNSHIP RECOMMENDATION
# ============================================================

def content_internship_scores(
    domain,
    experience,
    skills,
    interests
):

    user_profile = create_user_profile(
        domain,
        experience,
        skills,
        interests
    )

    internship_profiles = []

    for _, row in internships.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["experience_level"]).lower() + " "
            + str(row["required_skills"]).lower()
        )

        internship_profiles.append(profile)

    documents = [
        user_profile
    ] + internship_profiles

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    scores = {}

    for index, row in internships.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if (
            str(row["domain"]).lower()
            == str(domain).lower()
        ):
            score += 10

        # Experience bonus
        if (
            str(row["experience_level"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        scores[
            row["internship_id"]
        ] = score

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

    # Student does not exist
    if student_id not in rating_matrix.index:

        return {
            internship_id: 0
            for internship_id
            in internships["internship_id"]
        }

    # Student similarity
    student_similarity = cosine_similarity(
        rating_matrix
    )

    similarity_df = pd.DataFrame(
        student_similarity,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

    similar_students = similarity_df.loc[
        student_id
    ]

    # Remove current student
    similar_students = similar_students.drop(
        student_id,
        errors="ignore"
    )

    scores = {}

    for internship_id in internships[
        "internship_id"
    ]:

        weighted_sum = 0
        similarity_sum = 0

        for (
            other_student,
            similarity
        ) in similar_students.items():

            rating = rating_matrix.loc[
                other_student,
                internship_id
            ]

            if rating > 0:

                weighted_sum += (
                    similarity * rating
                )

                similarity_sum += similarity

        if similarity_sum > 0:

            predicted_rating = (
                weighted_sum
                / similarity_sum
            )

        else:

            predicted_rating = 0

        scores[
            internship_id
        ] = predicted_rating

    return scores


# ============================================================
# EVALUATION METRICS
# MAE + RMSE FOR COLLABORATIVE FILTERING
# ============================================================

def evaluate_collaborative_filtering():

    rating_matrix = ratings.pivot_table(
        index="student_id",
        columns="internship_id",
        values="rating",
        fill_value=0
    )

    actual_ratings = []
    predicted_ratings = []

    # Evaluate every known rating
    for _, rating_row in ratings.iterrows():

        student_id = rating_row[
            "student_id"
        ]

        internship_id = rating_row[
            "internship_id"
        ]

        actual_rating = float(
            rating_row["rating"]
        )

        # Create a copy for
        # leave-one-out evaluation
        evaluation_matrix = (
            rating_matrix.copy()
        )

        # Hide the rating being evaluated
        evaluation_matrix.loc[
            student_id,
            internship_id
        ] = 0

        # Calculate student similarity
        similarity_matrix = cosine_similarity(
            evaluation_matrix
        )

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=evaluation_matrix.index,
            columns=evaluation_matrix.index
        )

        similar_students = (
            similarity_df.loc[
                student_id
            ]
            .drop(
                student_id,
                errors="ignore"
            )
        )

        weighted_sum = 0
        similarity_sum = 0

        for (
            other_student,
            similarity
        ) in similar_students.items():

            other_rating = (
                evaluation_matrix.loc[
                    other_student,
                    internship_id
                ]
            )

            if (
                other_rating > 0
                and similarity > 0
            ):

                weighted_sum += (
                    similarity
                    * other_rating
                )

                similarity_sum += similarity

        # Only evaluate when
        # a prediction is possible
        if similarity_sum > 0:

            predicted_rating = (
                weighted_sum
                / similarity_sum
            )

            actual_ratings.append(
                actual_rating
            )

            predicted_ratings.append(
                predicted_rating
            )

    # No predictions available
    if not actual_ratings:

        return {
            "mae": 0,
            "rmse": 0
        }

    # Mean Absolute Error
    mae = mean_absolute_error(
        actual_ratings,
        predicted_ratings
    )

    # Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(
            actual_ratings,
            predicted_ratings
        )
    )

    return {
        "mae": mae,
        "rmse": rmse
    }


# ============================================================
# HYBRID INTERNSHIP RECOMMENDATION
# 60% CONTENT + 40% COLLABORATIVE
# ============================================================

def get_internship_recommendations(
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

        student_id = student.iloc[0][
            "student_id"
        ]

    else:

        # New user
        student_id = None

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content = content_internship_scores(
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
            for internship_id
            in internships["internship_id"]
        }

    # --------------------------------------------------------
    # Normalize collaborative scores
    # --------------------------------------------------------

    max_collaborative = (
        max(
            collaborative.values()
        )
        if collaborative
        else 0
    )

    normalized_collaborative = {}

    for (
        internship_id,
        score
    ) in collaborative.items():

        if max_collaborative > 0:

            normalized_score = (
                score
                / max_collaborative
            ) * 100

        else:

            normalized_score = 0

        normalized_collaborative[
            internship_id
        ] = normalized_score

    # --------------------------------------------------------
    # Hybrid scores
    # --------------------------------------------------------

    results = []

    for _, row in internships.iterrows():

        internship_id = row[
            "internship_id"
        ]

        content_score = content.get(
            internship_id,
            0
        )

        collaborative_score = (
            normalized_collaborative.get(
                internship_id,
                0
            )
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

            "duration":
                row["duration"],

            "location":
                row["location"],

            "score":
                hybrid_score
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
# PROJECT RECOMMENDATION
# ============================================================

def get_project_recommendations(
    domain,
    experience,
    skills,
    interests
):

    user_profile = create_user_profile(
        domain,
        experience,
        skills,
        interests
    )

    project_profiles = []

    for _, row in projects.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["difficulty"]).lower() + " "
            + str(row["required_skills"]).lower()
        )

        project_profiles.append(profile)

    documents = [
        user_profile
    ] + project_profiles

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    results = []

    for index, row in projects.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if (
            str(row["domain"]).lower()
            == str(domain).lower()
        ):
            score += 10

        # Difficulty / experience match
        if (
            str(row["difficulty"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        results.append({

            "title":
                row["title"],

            "domain":
                row["domain"],

            "required_skills":
                row["required_skills"],

            "difficulty":
                row["difficulty"],

            "score":
                score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# LEARNING RESOURCE RECOMMENDATION
# ============================================================

def get_resource_recommendations(
    domain,
    experience,
    skills,
    interests
):

    user_profile = create_user_profile(
        domain,
        experience,
        skills,
        interests
    )

    resource_profiles = []

    for _, row in resources.iterrows():

        profile = (
            str(row["domain"]).lower() + " "
            + str(row["level"]).lower() + " "
            + str(row["topics"]).lower()
        )

        resource_profiles.append(profile)

    documents = [
        user_profile
    ] + resource_profiles

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    results = []

    for index, row in resources.iterrows():

        score = similarity[index] * 100

        # Domain bonus
        if (
            str(row["domain"]).lower()
            == str(domain).lower()
        ):
            score += 10

        # Level / experience bonus
        if (
            str(row["level"]).lower()
            == str(experience).lower()
        ):
            score += 5

        score = min(score, 100)

        resource_id = row[
            "resource_id"
        ]

        results.append({

            "title":
                row["title"],

            "domain":
                row["domain"],

            "topics":
                row["topics"],

            "level":
                row["level"],

            "type":
                row["type"],

            "score":
                score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

evaluation_results = (
    evaluate_collaborative_filtering()
)


print()
print("============================================")
print("RECOMMENDATION ENGINE EVALUATION")
print("============================================")

print(
    f"Mean Absolute Error (MAE): "
    f"{evaluation_results['mae']:.4f}"
)

print(
    f"Root Mean Squared Error (RMSE): "
    f"{evaluation_results['rmse']:.4f}"
)

print("============================================")
print()


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

    # --------------------------------------------------------
    # Get form data
    # --------------------------------------------------------

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

    internship_results = (
        get_internship_recommendations(
            name,
            domain,
            experience,
            skills,
            interests
        )
    )

    # --------------------------------------------------------
    # Project recommendations
    # --------------------------------------------------------

    project_results = (
        get_project_recommendations(
            domain,
            experience,
            skills,
            interests
        )
    )

    # --------------------------------------------------------
    # Learning resources
    # --------------------------------------------------------

    resource_results = (
        get_resource_recommendations(
            domain,
            experience,
            skills,
            interests
        )
    )

    # --------------------------------------------------------
    # Send results to results.html
    # --------------------------------------------------------

    return render_template(

        "results.html",

        name=name,

        domain=domain,

        experience=experience,

        skills=skills,

        interests=interests,

        recommendations=internship_results,

        project_results=project_results,

        resource_results=resource_results

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)