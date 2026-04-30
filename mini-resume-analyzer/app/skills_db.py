def load_skills(path="data/skills_list.txt"):
    skills = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            skill = line.strip().lower()
            if skill:
                skills.append(skill)

    return list(set(skills))