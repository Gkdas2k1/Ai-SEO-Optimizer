# AI SEO Optimizer — Advanced Dashboard 🚀

![AI SEO Optimizer](https://img.shields.io/badge/AI-SEO-Optimizer-blue)

---

## Project Overview

The **AI SEO Optimizer** is an **interactive Streamlit application** that helps website owners, digital marketers, and SEO professionals analyze, optimize, and compare website content for search engines.  

Using **Gemini AI**, it can automatically **rewrite titles and meta descriptions** to improve SEO. Additionally, it provides **keyword extraction, clustering, SERP previews, competitor comparison**, and **interactive visual dashboards** for actionable insights.  

This tool is perfect for **data-driven SEO optimization** and makes it easy to generate professional reports in **CSV or PDF** format.  

---

## Features

1. **Website Analysis**  
   - Extracts title, meta description, headers, word count, and image alt tags.  
   - Provides an **SEO score** with recommendations for improvement.  

2. **Keyword Analysis**  
   - Extracts top keywords from website content.  
   - Clusters keywords using **TF-IDF** and **semantic embeddings** for actionable insights.  

3. **Competitor Comparison**  
   - Compare your website against a competitor URL.  
   - Identify keyword overlap, differences in headers, images, and word count.  

4. **AI-Powered Metadata Suggestions**  
   - Use **Gemini AI** to rewrite your title and meta description.  
   - Preview **before and after SERP snippets**.  

5. **Interactive Dashboard**  
   - **Venn diagram** for keyword overlap between your site and competitor.  
   - **Bar chart** of top keyword frequency.  
   - AI recommendations highlighted for easy reading.  

6. **Export Options**  
   - Export your SEO analysis and reports to **CSV** or **PDF**.  
   - Charts and AI suggestions included for presentations or client reporting.  

---

## Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/ai-seo-optimizer.git
cd ai-seo-optimizer
```

### 2️⃣ Create a Python 3.10 environment
```bash
python3.10 -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set your Gemini AI API key


#### Create a .env file in the project folder:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Usage
```
streamlit run app.py
```

- Enter your primary website URL.
- (Optional) Enter a competitor URL for comparison.
- Toggle advanced options in the sidebar.
- Generate AI metadata suggestions, keyword clusters, SERP previews, and visual charts.
- Export your analysis as CSV or PDF reports.

## Project Structure
```
ai-seo-optimizer/
│
├── app.py                # Main Streamlit application
├── seo_utils.py          # SEO extraction, analysis, clustering, AI functions
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignore venv, .env, cache, etc.
└── README.md             # Project description and instructions
```

## Technologies Used

- Python 3.10
- Streamlit – Interactive web application
- Gemini API – AI-powered content rewriting
- Matplotlib & matplotlib-venn – Dashboard visualizations
- Scikit-learn – Keyword clustering & TF-IDF
- Pandas – Data manipulation
- Fuzzywuzzy / NLP libraries – Text analysis
