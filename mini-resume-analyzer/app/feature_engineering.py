import re

from matplotlib import text

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if match:
        return match.group(0)
    return None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}", text)
    if match:
        return match.group(0)
    return None


def extract_links(text):
    links = []
    for match in re.findall(r"https?://[^\s]+", text):
        links.append(match)
    return links

def extract_skills(text, skills_list):
    found = []
    text_lower = text.lower()

    for skill in skills_list:
        if skill.lower() in text_lower:
            found.append(skill)

    return sorted(list(set(found)))

def estimate_experience_years(text):
    years = []
    for match in re.findall(r"(\d+)\s*\+?\s*years", text):
        years.append(int(match))
        
    for match in re.findall(r"(\d+)\s*months?", text):
        years.append(int(match) / 12)
    
    if years:
        return max(years)
    
    date_matches = re.findall(r"(19\d{2}|20\d{2})", text)
    date_matches = [int(x) for x in date_matches if 1950 <= int(x) <= 2026]

    if len(date_matches) >= 2:
        return max(date_matches) - min(date_matches)
    return 0

   
def cv_quality_score(text, skills_list=None):
    words = text.lower().split()
    word_count = len(words)
    score = 0
    
    if word_count < 50:
        return 0

    if 400 <= word_count <= 800:
        score += 25
    elif 200 <= word_count < 400 or 800 < word_count <= 1200:
        score += 15
    else:
        score += 5

    unique_ratio = len(set(words)) / word_count
    if unique_ratio > 0.6:
        score += 25
    elif unique_ratio > 0.4:
        score += 15

    action_words = ["increased", "decreased", "reduced", "saved", "maximized", "generated", "improved",
                    "optimized","led", "managed", "spearheaded", "orchestrated", "pioneered",
                       "mentored", "supervised", "coordinated","built", "developed", "implemented",
                        "designed", "architected", "engineered", "automated", "deployed","integrated",
                           "migrated", "configured", "debugged", "created", "programmed","maintained",
                            "supported", "monitored", "resolved", "facilitated", "streamlined", "revitalized"]  
    
    found_actions = sum(1 for w in words if w in action_words)
    score += min(found_actions * 4, 30)

    if skills_list:
        found_skills = sum(1 for skill in skills_list if skill.lower() in text.lower())
        score += min(found_skills * 2, 20)

    return score



