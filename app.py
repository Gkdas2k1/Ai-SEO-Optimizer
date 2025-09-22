import streamlit as st
from seo_utils import *
import seo_utils
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
import sys

# Redirect stderr to suppress gRPC warnings
if os.environ.get("DEV_MODE") != "true":
    sys.stderr = open(os.devnull, 'w')

st.set_page_config(page_title="AI SEO Optimizer", layout="wide")
st.title("🚀 AI-Powered SEO Optimizer (Gemini API + .env)")

url = st.text_input("Enter Website URL:")

if url:
    with st.spinner("🔎 Analyzing website..."):
        elements = seo_utils.extract_seo_elements(url)

    if "error" in elements:
        st.error(f"Error: {elements['error']}")
    else:
        st.subheader("📋 Extracted SEO Elements")
        st.json(elements)

        result = seo_utils.seo_score(elements)
        st.metric("SEO Score", f"{result['score']} / 100")

# ----------------------------
# Sidebar Options
# ----------------------------
st.sidebar.header("Options")
with st.sidebar.expander("Advanced features"):
    enable_competitor = st.checkbox("Enable Competitor Comparison", value=True)
    enable_clustering = st.checkbox("Enable Keyword Clustering (TF-IDF)", value=True)
    enable_semantic_clustering = st.checkbox("Semantic Keyword Clustering", value=True)
    top_k = st.number_input("Top keywords to extract", min_value=5, max_value=50, value=20)

# ----------------------------
# Inputs
# ----------------------------
url = st.text_input("Enter your Website URL (primary):", placeholder="https://example.com")
comp_url = st.text_input("Enter Competitor URL (optional):", placeholder="https://competitor.com")

primary = competitor = None
ai_meta = None

# ----------------------------
# Primary Site Analysis
# ----------------------------
if url:
    with st.spinner("🔎 Analyzing primary website..."):
        primary = extract_seo_elements(url)

    if "error" not in primary:
        st.subheader("📋 Primary Site - Extracted SEO Elements")
        st.json({k:v for k,v in primary.items() if k!="raw_text"})

        # SEO score
        result = seo_score(primary)
        st.metric("Primary SEO Score", f"{result['score']} / 100")
        if result["issues"]:
            st.subheader("⚠️ Primary Issues")
            for issue in result["issues"]: st.warning(issue)
        else:
            st.success("No major SEO issues found!")

        # Top keywords
        primary_keywords = extract_top_keywords([primary.get("raw_text","")], top_k=top_k)
        st.subheader("🔑 Primary Top Keywords")
        st.write(primary_keywords)

        # TF-IDF clustering
        if enable_clustering:
            st.subheader("🧩 Keyword Clusters (TF-IDF)")
            clusters = cluster_keywords(primary_keywords)
            for cid,kws in clusters.items():
                st.markdown(f"**Cluster {cid+1}:** {', '.join(kws)}")

        # Semantic clustering
        if enable_semantic_clustering:
            st.subheader("🧠 Semantic Keyword Clusters")
            sem_clusters = semantic_cluster_keywords(primary_keywords)
            for cid,kws in sem_clusters.items():
                st.markdown(f"**Cluster {cid+1}:** {', '.join(kws)}")

        # SERP preview
        st.subheader("🔍 SERP Snippet Preview (Current)")
        serp_current = make_serp_preview(primary.get("title",""), primary.get("description",""), primary.get("url",url))
        st.markdown(f"**{serp_current['display_title']}**")
        st.text(serp_current['display_url'])
        st.write(serp_current['display_description'])

        # AI rewrite
        st.subheader("✨ AI Suggestions (Gemini)")
        if st.button("Generate AI-Optimized Metadata (Gemini)"):
            keywords_for_prompt = primary_keywords[:5] if primary_keywords else ["SEO","optimization","web"]
            ai_meta = gemini_rewrite_metadata(primary.get("title",""), primary.get("description",""), keywords_for_prompt)
            st.success("AI suggestion generated")
            st.text(ai_meta)

        # Export buttons
        st.subheader("💾 Export Reports")
        if st.button("Export SEO Report to CSV"):
            csv_file = export_to_csv(primary, competitor if comp_url else None)
            st.success(f"Report exported: {csv_file}")

        if st.button("Export SEO Report to PDF"):
            pdf_file = export_to_pdf(primary, competitor if comp_url else None)
            st.success(f"Report exported: {pdf_file}")

# ----------------------------
# Competitor Analysis & Comparison
# ----------------------------
if enable_competitor and comp_url and primary:
    with st.spinner("🔎 Analyzing competitor..."):
        competitor = extract_seo_elements(comp_url)
    if "error" not in competitor:
        st.subheader("📊 Competitor Comparison")
        comparison = compare_sites(primary, competitor, top_k_keywords=top_k)
        if "error" not in comparison:
            st.metric("Keyword Overlap (%)", f"{comparison['overlap_score_percent']}%")
            st.write("Top keywords - Primary (top 10):", comparison["primary_top_keywords"])
            st.write("Top keywords - Competitor (top 10):", comparison["comp_top_keywords"])
            st.write("Overlapping keywords:", comparison["keyword_overlap"])
            st.write("---")
            st.write(f"Primary word count: {comparison['primary_word_count']}  |  Competitor word count: {comparison['comp_word_count']}")
            st.write(f"Primary headers: {comparison['primary_headers']}  |  Competitor headers: {comparison['comp_headers']}")
            st.write(f"Primary images w/ alt: {comparison['primary_images_alt']}  |  Competitor images w/ alt: {comparison['comp_images_alt']}")

            # ----------------------------
            # SEO Dashboard Visualizations
            # ----------------------------
            st.subheader("📈 SEO Dashboard Visualizations")

            # Keyword overlap Venn diagram
            primary_set = set(comparison["primary_top_keywords"])
            comp_set = set(comparison["comp_top_keywords"])
            st.write("### Keyword Overlap Venn Diagram")
            fig, ax = plt.subplots(figsize=(5,5))
            venn2([primary_set, comp_set], set_labels=("Primary", "Competitor"))
            st.pyplot(fig)

            # Top keyword frequency bar chart
            st.write("### Top Keywords Frequency (Primary)")
            kws = comparison["primary_top_keywords"]
            counts = [primary["raw_text"].lower().count(k.lower()) for k in kws]
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.barh(kws[::-1], counts[::-1], color="skyblue")
            ax2.set_xlabel("Frequency in Content")
            st.pyplot(fig2)

            # AI SERP Recommendation highlights
            if ai_meta:
                st.subheader("✨ AI Optimized Metadata Preview")
                st.write("**Before:**")
                st.text(f"Title: {primary['title']}\nDescription: {primary['description']}")
                st.write("**After (AI Recommendation):**")
                st.text(ai_meta)

# Keyword density analysis
st.subheader("🔑 Keyword Density")
if url:
    with st.spinner("Analyzing keyword density..."):
        primary = seo_utils.extract_seo_elements(url)
        if "error" not in primary:
            text_content = " ".join(primary["headers"]) + " " + primary["description"]
            keyword_density = seo_utils.analyze_keyword_density(text_content)
            st.bar_chart(keyword_density)
        else:
            st.warning("Could not analyze keyword density due to an error.")

# Readability score
st.subheader("📖 Readability Score (Flesch-Kincaid)")
if url:
    with st.spinner("Analyzing readability..."):
        primary = seo_utils.extract_seo_elements(url)
        if "error" not in primary:
            text_content = " ".join(primary["headers"]) + " " + primary["description"]
            readability_scores = seo_utils.get_readability_scores(text_content)
            st.json(readability_scores)
        else:
            st.warning("Could not analyze readability due to an error.")

# Mobile-friendliness check
st.subheader("📱 Mobile-Friendliness")
if url:
    with st.spinner("Checking mobile-friendliness..."):
        mobile_friendly = seo_utils.check_mobile_friendliness(url)
        if mobile_friendly:
            st.success("Page appears to be mobile-friendly.")
        else:
            st.warning("Page may not be mobile-friendly.")

# Performance metrics
st.subheader("⚡️ Performance Metrics")
if url:
    with st.spinner("Analyzing performance..."):
        performance = seo_utils.get_performance_metrics(url)
        if performance:
            st.json(performance)
        else:
            st.warning("Could not retrieve performance metrics.")

# Social media meta tags
st.subheader("💬 Social Media Meta Tags")
if url:
    with st.spinner("Checking social media tags..."):
        social_tags = seo_utils.get_social_media_tags(url)
        if social_tags:
            st.json(social_tags)
        else:
            st.info("No social media meta tags found.")

# PDF report generation
st.subheader("📄 Download SEO Report")
if url:
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF..."):
            primary = seo_utils.extract_seo_elements(url)
            if "error" not in primary:
                report_filename = seo_utils.generate_seo_report(url, primary)
                with open(report_filename, "rb") as f:
                    st.download_button("Download Report", f, file_name=report_filename)
            else:
                st.error("Could not generate report due to an error.")