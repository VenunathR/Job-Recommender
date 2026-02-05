import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- EXTRA DOMAIN STOPWORDS ----------
extra_stopwords = {
    "looking","responsible","team","work","skills",
    "good","communication","ability","knowledge","year","years",
    "requirements","job","role","candidate","company","using",
    "development","support","strong"
}

stop_words = ENGLISH_STOP_WORDS.union(extra_stopwords)

# ---------- SKILL KEYWORDS TO BOOST ----------
skill_keywords = {
    "python","java","sql","machine","learning","deep","nlp",
    "aws","docker","kubernetes","tensorflow","pytorch",
    "pandas","numpy","flask","api","data","analysis",
    "spark","hadoop","tableau","powerbi","excel"
}

# ---------- LOAD DATA ----------
df = pd.read_csv("data/jobs.csv")

df['text'] = df['Job Title'].fillna('') + " " + df['Job Description'].fillna('')

# ---------- CLEAN FUNCTION ----------
def clean_text(t):
    t = t.lower()
    t = re.sub(r'[^a-zA-Z ]', ' ', t)
    words = [w for w in t.split() if w not in stop_words and len(w) > 2]

    # Boost skill terms
    boosted_words = []
    for w in words:
        boosted_words.append(w)
        if w in skill_keywords:
            boosted_words.append(w)

    return " ".join(boosted_words)

df['text'] = df['text'].apply(clean_text)

df.to_csv("data/cleaned_jobs.csv", index=False)

print("Preprocessing complete using sklearn stopwords.")
