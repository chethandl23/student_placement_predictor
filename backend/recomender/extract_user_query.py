# backend/recommender/extract_user_query.py
import os
import json
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment. Put it in .env")

# Import Groq client
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

def _simple_fallback_parse(user_input: str):
    """
    Lightweight fallback parsing (if API fails): attempts to extract role and comma-separated skills.
    """
    text = user_input.lower()
    # naive role detection (look for known roles)
    possible_roles = [
        "data scientist", "data analyst", "machine learning engineer",
        "machine learning", "software developer", "full stack developer",
        "cloud engineer", "devops engineer", "cybersecurity", "ai engineer", "ui/ux designer"
    ]
    role = None
    for r in possible_roles:
        if r in text:
            role = r
            break

    # attempt to find skills after "i know" or "skills are"
    skills = []
    import re
    m = re.search(r"(i know|skills are|my skills are|i have experience in)(.*)", text)
    if m:
        tail = m.group(2)
        # split on commas or 'and'
        parts = re.split(r",|and|;", tail)
        for p in parts:
            p = p.strip()
            if len(p) > 1:
                # keep first two words max as skill phrase
                skill = " ".join(p.split()[:3])
                skills.append(skill)
    return role, skills

def extract_user_info(user_input: str):
    """
    Uses Groq chat completion to extract JSON {job_role, skills}.
    Returns (role, skills_list) or (None, []) on failure.
    """
    prompt = f"""
Extract the intended job role and the skills the user already knows from the following short text.
Return ONLY valid JSON in this exact format (no extra commentary):
{{"job_role": "role here", "skills": ["skill1", "skill2"]}}

Text:
\"\"\"{user_input}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256
        )
        text = response.choices[0].message.content.strip()
        # try to parse JSON strictly
        parsed = json.loads(text)
        job_role = parsed.get("job_role")
        skills = parsed.get("skills", [])
        # normalize
        if job_role:
            job_role = job_role.lower().strip()
        skills = [s.lower().strip() for s in skills if isinstance(s, str) and s.strip()]
        return job_role, skills
    except Exception as e:
        # print exception for debugging (optional)
        print("Groq extraction failed:", e)
        # fallback parsing
        role, skills = _simple_fallback_parse(user_input)
        return role, skills or []
