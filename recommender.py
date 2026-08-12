import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========================================
# LOAD DATA
# ========================================

students = pd.read_csv("students.csv")
internships = pd.read_csv("internships.csv")
ratings = pd.read_csv("ratings.csv")
projects = pd.read_csv("projects.csv")
resources = pd.read_csv("resources.csv")


# ========================================
# DISPLAY DATA
# ========================================

print("Students:")
print(students)

print("\nInternships:")
print(internships)

print("\nRatings:")
print(ratings)


# ========================================
# SELECT STUDENT
# ========================================

student_id = "S001"

student = students[
    students["student_id"] == student_id
].iloc[0]


# ========================================
# CONTENT-BASED RECOMMENDATION
# ========================================

student_profile = (
    student["skills"]
    + " "
    + student["domain"]
    + " "
    + student["interests"]
)

internship_profiles = (
    internships["required_skills"]
    + " "
    + internships["domain"]
)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    [student_profile] + internship_profiles.tolist()
)

similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)[0]

internships["content_score"] = similarity_scores

content_recommendations = internships.sort_values(
    "content_score",
    ascending=False
)


print("\n========================================")
print("CONTENT-BASED RECOMMENDATIONS")
print("========================================")

print(f"Student: {student['name']}")
print(f"Domain: {student['domain']}")
print(f"Skills: {student['skills']}")

print("\nTop Recommendations:")

for _, recommendation in content_recommendations.head(5).iterrows():

    print(f"\n{recommendation['title']}")
    print(f"Company: {recommendation['company']}")
    print(f"Domain: {recommendation['domain']}")

    print(
        f"Match Score: "
        f"{recommendation['content_score'] * 100:.2f}%"
    )


# ========================================
# RATING MATRIX
# ========================================

rating_matrix = ratings.pivot_table(
    index="student_id",
    columns="internship_id",
    values="rating",
    fill_value=0
)

print("\n========================================")
print("RATING MATRIX")
print("========================================")

print(rating_matrix)


# ========================================
# COLLABORATIVE FILTERING
# ========================================

student_vector = rating_matrix.loc[student_id]

similarities = cosine_similarity(
    rating_matrix,
    rating_matrix.loc[[student_id]]
).flatten()

similar_students = pd.Series(
    similarities,
    index=rating_matrix.index
)

similar_students = similar_students.drop(
    student_id
)

similar_students = similar_students.sort_values(
    ascending=False
)


print("\n========================================")
print("SIMILAR STUDENTS")
print("========================================")

print(similar_students)


# ========================================
# COLLABORATIVE RECOMMENDATIONS
# ========================================

collaborative_scores = {}

for similar_student_id, similarity in similar_students.items():

    similar_student_ratings = rating_matrix.loc[
        similar_student_id
    ]

    for internship_id, rating in similar_student_ratings.items():

        # Recommend only internships
        # that the current student has not rated

        if (
            student_vector[internship_id] == 0
            and rating > 0
        ):

            if internship_id not in collaborative_scores:
                collaborative_scores[internship_id] = 0

            collaborative_scores[internship_id] += (
                similarity * rating
            )


collaborative_recommendations = sorted(
    collaborative_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


print("\n========================================")
print("COLLABORATIVE RECOMMENDATIONS")
print("========================================")

for internship_id, score in collaborative_recommendations[:5]:

    internship = internships[
        internships["internship_id"] == internship_id
    ].iloc[0]

    print(f"\n{internship['title']}")
    print(f"Company: {internship['company']}")
    print(f"Collaborative Score: {score:.2f}")


# ========================================
# HYBRID RECOMMENDATION
# ========================================

collaborative_score_dict = dict(
    collaborative_recommendations
)

if collaborative_score_dict:

    max_collaborative_score = max(
        collaborative_score_dict.values()
    )

else:

    max_collaborative_score = 1


hybrid_results = []


for _, internship in internships.iterrows():

    internship_id = internship["internship_id"]

    content_score = internship["content_score"]

    collaborative_score = (
        collaborative_score_dict.get(
            internship_id,
            0
        )
    )

    normalized_collaborative = (
        collaborative_score
        /
        max_collaborative_score
    )

    # 60% content-based
    # 40% collaborative filtering

    hybrid_score = (
        0.6 * content_score
        +
        0.4 * normalized_collaborative
    )

    hybrid_results.append({
        "internship_id": internship_id,
        "title": internship["title"],
        "company": internship["company"],
        "domain": internship["domain"],
        "content_score": content_score,
        "collaborative_score": normalized_collaborative,
        "hybrid_score": hybrid_score
    })


hybrid_df = pd.DataFrame(
    hybrid_results
)

hybrid_df = hybrid_df.sort_values(
    "hybrid_score",
    ascending=False
)


print("\n========================================")
print("HYBRID RECOMMENDATIONS")
print("========================================")


for _, recommendation in hybrid_df.head(5).iterrows():

    print(f"\n{recommendation['title']}")
    print(f"Company: {recommendation['company']}")
    print(f"Domain: {recommendation['domain']}")

    print(
        f"Content Score: "
        f"{recommendation['content_score'] * 100:.2f}%"
    )

    print(
        f"Collaborative Score: "
        f"{recommendation['collaborative_score'] * 100:.2f}%"
    )

    print(
        f"Final Hybrid Score: "
        f"{recommendation['hybrid_score'] * 100:.2f}%"
    )


# ========================================
# CLEAN FINAL INTERNSHIP RECOMMENDATIONS
# ========================================

print("\n")
print("=" * 55)
print("       AI INTERNSHIP RECOMMENDATION ENGINE")
print("=" * 55)

print(f"\nStudent: {student['name']}")
print(f"Domain: {student['domain']}")
print(
    f"Experience Level: "
    f"{student['experience_level']}"
)
print(f"Skills: {student['skills']}")

print("\nTOP 5 INTERNSHIP RECOMMENDATIONS")
print("-" * 55)


for rank, (_, recommendation) in enumerate(
    hybrid_df.head(5).iterrows(),
    start=1
):

    print(
        f"\n{rank}. "
        f"{recommendation['title']}"
    )

    print(
        f"   Company: "
        f"{recommendation['company']}"
    )

    print(
        f"   Domain: "
        f"{recommendation['domain']}"
    )

    print(
        f"   Match Score: "
        f"{recommendation['hybrid_score'] * 100:.2f}%"
    )


print("\n" + "-" * 55)

print("Recommendation Method:")
print(
    "Content-Based Filtering + "
    "Collaborative Filtering"
)

print(
    "Hybrid Weight: "
    "60% Content / 40% Collaborative"
)

print("=" * 55)


# ========================================
# PROJECT RECOMMENDATIONS
# ========================================

print("\n")
print("=" * 55)
print("             PROJECT RECOMMENDATIONS")
print("=" * 55)


student_skills = set(
    skill.strip().lower()
    for skill in student["skills"].split(",")
)


project_results = []


for _, project in projects.iterrows():

    required_skills = set(
        skill.strip().lower()
        for skill in project["required_skills"].split(",")
    )

    if required_skills:

        matched_skills = (
            student_skills.intersection(
                required_skills
            )
        )

        match_score = (
            len(matched_skills)
            /
            len(required_skills)
        )

    else:

        match_score = 0


    # Domain matching bonus

    if (
        project["domain"].lower()
        ==
        student["domain"].lower()
    ):

        match_score += 0.2


    match_score = min(
        match_score,
        1.0
    )


    project_results.append({
        "project_id": project["project_id"],
        "title": project["title"],
        "domain": project["domain"],
        "difficulty": project["difficulty"],
        "match_score": match_score
    })


project_df = pd.DataFrame(
    project_results
)

project_df = project_df.sort_values(
    "match_score",
    ascending=False
)


print(
    f"\nRecommended projects for "
    f"{student['name']}:"
)

print("-" * 55)


for rank, (_, project) in enumerate(
    project_df.head(5).iterrows(),
    start=1
):

    print(
        f"\n{rank}. "
        f"{project['title']}"
    )

    print(
        f"   Domain: "
        f"{project['domain']}"
    )

    print(
        f"   Difficulty: "
        f"{project['difficulty']}"
    )

    print(
        f"   Match Score: "
        f"{project['match_score'] * 100:.2f}%"
    )


# ========================================
# LEARNING RESOURCE RECOMMENDATIONS
# ========================================

print("\n")
print("=" * 55)
print("        LEARNING RESOURCE RECOMMENDATIONS")
print("=" * 55)


student_skills = set(
    skill.strip().lower()
    for skill in student["skills"].split(",")
)

student_interests = set(
    interest.strip().lower()
    for interest in student["interests"].split(",")
)


resource_results = []


for _, resource in resources.iterrows():

    resource_topics = set(
        topic.strip().lower()
        for topic in resource["topics"].split(",")
    )


    # Skills matching

    skill_matches = (
        student_skills.intersection(
            resource_topics
        )
    )


    # Interest matching

    interest_matches = (
        student_interests.intersection(
            resource_topics
        )
    )


    if resource_topics:

        skill_score = (
            len(skill_matches)
            /
            len(resource_topics)
        )

        interest_score = (
            len(interest_matches)
            /
            len(resource_topics)
        )

    else:

        skill_score = 0
        interest_score = 0


    # 70% skills + 30% interests

    match_score = (
        0.7 * skill_score
        +
        0.3 * interest_score
    )


    # Domain bonus

    if (
        resource["domain"].lower()
        ==
        student["domain"].lower()
    ):

        match_score += 0.2


    match_score = min(
        match_score,
        1.0
    )


    resource_results.append({
        "resource_id": resource["resource_id"],
        "title": resource["title"],
        "domain": resource["domain"],
        "level": resource["level"],
        "type": resource["type"],
        "match_score": match_score
    })


resource_df = pd.DataFrame(
    resource_results
)

resource_df = resource_df.sort_values(
    "match_score",
    ascending=False
)


print(
    f"\nRecommended learning resources for "
    f"{student['name']}:"
)

print("-" * 55)


for rank, (_, resource) in enumerate(
    resource_df.head(5).iterrows(),
    start=1
):

    print(
        f"\n{rank}. "
        f"{resource['title']}"
    )

    print(
        f"   Domain: "
        f"{resource['domain']}"
    )

    print(
        f"   Level: "
        f"{resource['level']}"
    )

    print(
        f"   Type: "
        f"{resource['type']}"
    )

    print(
        f"   Match Score: "
        f"{resource['match_score'] * 100:.2f}%"
    )


# ========================================
# END
# ========================================

print("\n")
print("=" * 55)
print("              END OF RECOMMENDATIONS")
print("=" * 55)

def get_recommendations(name, domain, experience, skills, interests):
    """
    Generate internship, project, and learning-resource recommendations.
    """

    # Convert user skills/interests into lowercase sets
    user_skills = set(
        skill.strip().lower()
        for skill in skills.split(",")
        if skill.strip()
    )

    user_interests = set(
        interest.strip().lower()
        for interest in interests.split(",")
        if interest.strip()
    )

    internship_results = []

    # Calculate a simple content-based match
    for _, internship in internships.iterrows():

        required_skills = set(
            skill.strip().lower()
            for skill in internship["required_skills"].split(",")
            if skill.strip()
        )

        if required_skills:
            skill_match = (
                len(user_skills & required_skills)
                / len(required_skills)
            )
        else:
            skill_match = 0

        domain_match = (
            1 if domain.lower() == internship["domain"].lower()
            else 0
        )

        experience_match = (
            1 if experience.lower()
            == internship["experience_level"].lower()
            else 0
        )

        score = (
            skill_match * 0.6
            + domain_match * 0.25
            + experience_match * 0.15
        )

        internship_results.append({
            "title": internship["title"],
            "company": internship["company"],
            "domain": internship["domain"],
            "score": round(score * 100, 2),
            "duration": internship["duration"],
            "location": internship["location"]
        })

    internship_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "internships": internship_results[:5],
        "projects": [],
        "learning_resources": []
    }