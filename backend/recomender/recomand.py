import re
from backend.recomender.courses_data import job_roles_skills, courses_resources

def clean_skill_text(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def recommend_courses(job_role, user_skills):
    required_skills = job_roles_skills.get(job_role, [])

    # Clean user skills (remove punctuation & lower)
    cleaned_user_skills = [clean_skill_text(s) for s in user_skills]

    missing_skills = [
        skill for skill in required_skills 
        if clean_skill_text(skill) not in cleaned_user_skills
    ]

    recommendations = []
    for skill in missing_skills:
        if skill in courses_resources:
            for course_name, course_link in courses_resources[skill]:
                recommendations.append({
                    "skill": skill,
                    "course": course_name,
                    "url": f"https://{course_link}"
                })

    return missing_skills, recommendations[:10]
