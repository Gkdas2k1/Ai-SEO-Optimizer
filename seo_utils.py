import os
import io
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from collections import Counter
from datetime import datetime
import textstat
import time
from datetime import datetime
import time

# Load environment variables and configure Gemini
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# ----------------------------
# Web scraping & SEO elements
# ----------------------------
def safe_get_text(soup):
    return soup.get_text(separator=" ", strip=True)

def extract_seo_elements(url: str):
    """Extract SEO elements from a URL"""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "lxml")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag["content"].strip() if description_tag and description_tag.get("content") else "No Meta Description"

        headers = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3"])]
        images = [img.get("alt") for img in soup.find_all("img") if img.get("alt")]
        text = safe_get_text(soup)
        word_count = len(re.findall(r"\w+", text))

        return {
            "url": url,
            "title": title,
            "description": description,
            "headers": headers,
            "images_alt": images,
            "word_count": word_count,
            "raw_text": text
        }
    except Exception as e:
        return {"error": str(e), "url": url}

def seo_score(elements: dict):
    """Evaluates SEO elements and assigns a score with issues list"""
    score = 0
    issues = []

    if "error" in elements:
        return {"score":0, "issues":[elements.get("error","Unknown error")]}

    if 50 <= len(elements["title"]) <= 60: score += 20
    else: issues.append("Title length not optimal (50–60 chars).")

    if 150 <= len(elements["description"]) <= 160: score += 20
    else: issues.append("Meta description length not optimal (150–160 chars).")

    if elements["word_count"] >= 300: score += 20
    else: issues.append("Content too short (<300 words).")

    if elements["headers"]: score += 20
    else: issues.append("No headers found (H1/H2/H3).")

    if elements["images_alt"]: score += 20
    else: issues.append("Images missing alt text.")

    return {"score":score, "issues":issues}

# ----------------------------
# Gemini AI rewriting
# ----------------------------
def gemini_rewrite_metadata(title, description, keywords):
    """Uses Gemini AI to rewrite SEO metadata"""
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)
    
    prompt = f"""
    You are an SEO expert. Rewrite the following metadata to be keyword-rich,
    SEO optimized, and engaging but natural. Keep the title within 50-60 characters
    and description around 150-160 characters. Provide the result in JSON-like form:

    Title: {title}
    Description: {description}
    Target Keywords: {', '.join(keywords)}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if response and response.text else "No AI suggestion generated."

def analyze_keyword_density(text: str):
    """Analyzes keyword density of a given text"""
    words = re.findall(r'\w+', text.lower())
    # Simple stopword list, can be expanded
    stopwords = set(["the", "a", "an", "in", "is", "it", "and", "of", "to", "for", "on", "with", "as", "by", "that", "this"])
    filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
    word_counts = Counter(filtered_words)
    total_words = len(filtered_words)
    density = {word: (count / total_words) * 100 for word, count in word_counts.most_common(10)}
    return density

def get_readability_scores(text: str):
    """Calculates readability scores for a given text"""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
    }

def check_mobile_friendliness(url: str):
    """Checks for mobile-friendliness by looking for a viewport meta tag"""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "lxml")
        viewport_tag = soup.find("meta", attrs={"name": "viewport"})
        return viewport_tag is not None
    except Exception:
        return False

def get_performance_metrics(url: str):
    """Gets basic performance metrics for a URL"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        end_time = time.time()
        return {
            "response_time_seconds": round(end_time - start_time, 2),
            "status_code": response.status_code,
        }
    except Exception:
        return None

def get_social_media_tags(url: str):
    """Extracts social media meta tags (Open Graph and Twitter)"""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "lxml")
        social_tags = {}
        for tag in soup.find_all("meta"):
            prop = tag.get("property") or tag.get("name")
            if prop and (prop.startswith("og:") or prop.startswith("twitter:")):
                social_tags[prop] = tag.get("content")
        return social_tags if social_tags else None
    except Exception:
        return None

def generate_seo_report(url: str, elements: dict):
    """Generates a PDF SEO report"""
    filename = f"seo_report_{url.replace('https://', '').replace('http://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, f"SEO Report for: {url}")
    c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    c.drawString(100, 700, "--- Basic SEO Elements ---")
    c.drawString(120, 680, f"Title: {elements.get('title', 'N/A')}")
    c.drawString(120, 660, f"Description: {elements.get('description', 'N/A')[:100]}") # Truncate long description
    c.drawString(120, 640, f"Word Count: {elements.get('word_count', 'N/A')}")

    score_data = seo_score(elements)
    c.drawString(100, 610, "--- SEO Score ---")
    c.drawString(120, 590, f"Score: {score_data['score']} / 100")
    
    c.drawString(100, 560, "--- Issues ---")
    y = 540
    for issue in score_data['issues']:
        c.drawString(120, y, f"- {issue}")
        y -= 20
        if y < 50: # new page
            c.showPage()
            y = 750

    c.save()
    return filename

# ----------------------------
# Keyword extraction & clustering
# ----------------------------
def extract_top_keywords(texts, top_k=20, ngram_range=(1,2)):
    """Extract top TF-IDF keywords"""
    if not texts: return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
    top_indices = tfidf_sum.argsort()[::-1][:top_k]
    return feature_array[top_indices].tolist()

def cluster_keywords(keywords, n_clusters=None):
    """Simple TF-IDF KMeans clustering"""
    if not keywords: return {}
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer="char_wb")
    X = vec.fit_transform(keywords)
    if n_clusters is None: n_clusters = min(5, max(1, len(keywords)//4))
    n_clusters = min(n_clusters, len(keywords))
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    clusters = {}
    for kw, lab in zip(keywords, labels):
        clusters.setdefault(int(lab), []).append(kw)
    return clusters

def semantic_cluster_keywords(keywords, n_clusters=None):
    """Semantic clustering using Sentence Transformers"""
    if not keywords: return {}
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(keywords)
    if n_clusters is None: n_clusters = min(5, max(1, len(keywords)//4))
    n_clusters = min(n_clusters, len(keywords))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for kw, lab in zip(keywords, labels):
        clusters.setdefault(int(lab), []).append(kw)
    return clusters

# ----------------------------
# Competitor comparison
# ----------------------------
def compare_sites(primary_elements, competitor_elements, top_k_keywords=20):
    """Compare two websites and calculate keyword overlap + metrics"""
    if "error" in primary_elements or "error" in competitor_elements:
        return {"error":"One of the sites could not be fetched."}
    
    def build_doc(el):
        parts = [el.get("title",""), el.get("description",""), " ".join(el.get("headers",[])), el.get("raw_text","")]
        return " ".join([p for p in parts if p])

    primary_doc = build_doc(primary_elements)
    comp_doc = build_doc(competitor_elements)
    combined = [primary_doc, comp_doc]

    top_keywords = extract_top_keywords(combined, top_k=top_k_keywords)
    primary_top = extract_top_keywords([primary_doc], top_k=top_k_keywords)
    comp_top = extract_top_keywords([comp_doc], top_k=top_k_keywords)

    overlap = list(set(primary_top).intersection(set(comp_top)))
    overlap_score = round(100 * len(overlap) / max(1, min(len(primary_top), len(comp_top))),2)

    metrics = {
        "primary_word_count": primary_elements.get("word_count",0),
        "comp_word_count": competitor_elements.get("word_count",0),
        "primary_headers": len(primary_elements.get("headers",[])),
        "comp_headers": len(competitor_elements.get("headers",[])),
        "primary_images_alt": len(primary_elements.get("images_alt",[])),
        "comp_images_alt": len(competitor_elements.get("images_alt",[])),
        "keyword_overlap": overlap,
        "overlap_score_percent": overlap_score,
        "primary_top_keywords": primary_top[:10],
        "comp_top_keywords": comp_top[:10]
    }
    return metrics

# ----------------------------
# SERP snippet preview
# ----------------------------
def make_serp_preview(title, description, display_url):
    """Build a simple SERP-like preview"""
    def trim_text(txt,max_chars):
        if len(txt) <= max_chars: return txt
        cut = txt[:max_chars].rfind(" ")
        if cut==-1: cut=max_chars
        return txt[:cut]+"…"
    return {
        "display_title": trim_text(title,60),
        "display_url": display_url,
        "display_description": trim_text(description,160)
    }

# ----------------------------
# Export functions
# ----------------------------
def export_to_csv(primary, competitor=None, filename="seo_report.csv"):
    rows=[]
    def add_site_data(name,elements):
        if "error" in elements: return
        rows.append({
            "site":name,
            "url":elements.get("url"),
            "title":elements.get("title"),
            "description":elements.get("description"),
            "word_count":elements.get("word_count"),
            "headers":", ".join(elements.get("headers",[])),
            "images_alt_count":len(elements.get("images_alt",[]))
        })
    add_site_data("Primary",primary)
    if competitor: add_site_data("Competitor",competitor)
    df=pd.DataFrame(rows)
    df.to_csv(filename,index=False)
    return filename

def export_to_pdf(primary, competitor=None, filename="seo_report.pdf"):
    buffer=io.BytesIO()
    c=canvas.Canvas(buffer,pagesize=letter)
    width,height=letter
    y=height-50
    def add_site_data(name,elements):
        nonlocal y
        if "error" in elements: return
        c.setFont("Helvetica-Bold",14)
        c.drawString(50,y,f"{name} Site SEO Report"); y-=20
        c.setFont("Helvetica",12)
        c.drawString(50,y,f"URL: {elements.get('url')}"); y-=15
        c.drawString(50,y,f"Title: {elements.get('title')}"); y-=15
        c.drawString(50,y,f"Description: {elements.get('description')}"); y-=15
        c.drawString(50,y,f"Word Count: {elements.get('word_count')}"); y-=15
        c.drawString(50,y,f"Headers: {', '.join(elements.get('headers',[]))}"); y-=15
        c.drawString(50,y,f"Images with alt text: {len(elements.get('images_alt',[]))}"); y-=40
    add_site_data("Primary",primary)
    if competitor: add_site_data("Competitor",competitor)
    c.save()
    buffer.seek(0)
    with open(filename,"wb") as f: f.write(buffer.read())
    return filename