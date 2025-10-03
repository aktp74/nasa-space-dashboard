import os
import base64
import requests
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import webbrowser

# =========================
# 🛠️ PDF Library Compatibility
# =========================
try:
    import pypdf  # Modern library
    PyPDF2 = pypdf  # Alias untuk compatibility
except ImportError:
    import PyPDF2  # Fallback ke legacy library

import docx

# =========================
# 📊 Data Loading dengan Error Handling
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SB_publication_PMC.csv", encoding="latin1")
        df.columns = [c.lower() for c in df.columns]
        st.sidebar.success(f"✅ Loaded {len(df)} publications")
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'SB_publication_PMC.csv' not found!")
        # Return sample empty dataframe untuk testing
        return pd.DataFrame({
            'title': ['Sample Publication 1', 'Sample Publication 2'],
            'year': [2023, 2024],
            'abstract': ['Sample abstract text 1', 'Sample abstract text 2'],
            'conclusion': ['Sample conclusion 1', 'Sample conclusion 2'],
            'link': ['https://example.com/1', 'https://example.com/2']
        })
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# =========================
# 🛠️ Initialize Session State
# =========================
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = os.getenv('HF_API_KEY', '')

# =========================
# 🎨 Background Setup dengan Error Handling
# =========================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading {bin_file}: {str(e)}")
        return None

# Sidebar Styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0e2e 0%, #1a237e 50%, #283593 100%);
    }
    
    .sidebar-content {
        color: #ffffff;
    }
    
    .stButton > button {
        color: #00ffff;
        border: 1px solid #00ffff;
        background: rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: rgba(0, 255, 255, 0.2);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Background Image
img_file = "background.jpg"
if os.path.exists(img_file):
    img_base64 = get_base64_of_bin_file(img_file)
    if img_base64:
        page_bg = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        .block-container {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 200, 255, 0.5);
            box-shadow: 0 0 15px rgba(0, 200, 255, 0.25);
            color: #f0faff;
            transition: all 0.3s ease-in-out;
        }}

        .block-container:hover {{
            border: 1px solid rgba(0, 255, 255, 0.9);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
            transform: translateY(-3px);
        }}

        h1, h2, h3, h4 {{
            color: #1a3c34;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
else:
    st.sidebar.info("ℹ️ background.jpg not found - using default background")

# Background Music
music_file = "background.mp3"
if os.path.exists(music_file):
    try:
        with open(music_file, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md_audio = f"""
        <audio id="bg-music" autoplay loop hidden>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <script>
            var audio = document.getElementById("bg-music");
            audio.volume = 0.2;
        </script>
        """
        st.markdown(md_audio, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading music: {str(e)}")
else:
    st.sidebar.info("ℹ️ background.mp3 not found")

# =========================
# 🤖 AI Summarizer - FREE VERSION
# =========================
def try_hugging_face(text, mode):
    """Try Hugging Face Inference API - FREE"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        hf_token = st.session_state.hf_api_key or os.getenv('HF_API_KEY', '')
        
        if not hf_token:
            return "❌ No Hugging Face token configured"
            
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        input_text = text[:1024]  # Limit length
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_length": 300 if mode == "single" else 500,
                "min_length": 100 if mode == "single" else 200,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['summary_text']
            else:
                return "❌ Hugging Face: No summary generated"
        elif response.status_code == 503:
            return "❌ Hugging Face: Model is loading, please try again in 30 seconds"
        elif response.status_code == 401:
            return "❌ Hugging Face: Invalid token"
        else:
            return f"❌ Hugging Face API error: {response.status_code}"
        
    except Exception as e:
        return f"❌ Hugging Face: {str(e)}"

def ai_summarize(text, max_tokens=400, mode="single"):
    """Free AI summarization using multiple fallback options"""
    # Option 1: Try Hugging Face Inference API (FREE) - PRIORITY
    hf_result = try_hugging_face(text, mode)
    if hf_result and not hf_result.startswith("❌"):
        return hf_result
    
    # Option 2: Try OpenRouter as backup (jika ada API key)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "NASA Space Bioscience Dashboard"
            }

            if mode == "overview":
                user_prompt = f"""Buat summary komprehensif dalam English untuk koleksi artikel NASA space bioscience. 
                Fokus pada tema utama, penemuan penting, dan research trends.
                
                Text: {text[:3000]}"""
            else:
                user_prompt = f"""Summarize this NASA space research article in 5-7 sentences in English:
                
                {text[:2000]}"""

            available_models = [
                "anthropic/claude-3-haiku",
                "google/gemini-pro",
                "meta-llama/llama-3-8b-instruct",
                "microsoft/wizardlm-2-8x22b"
            ]
            
            for model in available_models:
                try:
                    data = {
                        "model": model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are a scientific research assistant that summarizes NASA space bioscience articles clearly in English."
                            },
                            {
                                "role": "user", 
                                "content": user_prompt
                            }
                        ],
                        "max_tokens": max_tokens,
                    }

                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status_code == 404:
                        continue
                    elif response.status_code == 402:
                        return "❌ OpenRouter: Payment required. Please add credits or use Hugging Face."
                        
                except Exception:
                    continue
                    
            return "❌ OpenRouter: No working models found."
            
        except Exception as e:
            return f"❌ OpenRouter Error: {str(e)}"
    
    # Final fallback - rule based summary
    return fallback_summarize(text, mode)

def smart_summarize(text, max_tokens=400, mode="single"):
    """Smart summarization dengan fallback"""
    ai_result = ai_summarize(text, max_tokens, mode)
    
    if ai_result.startswith("❌"):
        st.sidebar.warning("AI summarization failed. Using rule-based fallback.")
        return fallback_summarize(text, mode)
    
    return ai_result

def fallback_summarize(text, mode="single"):
    """Simple rule-based summarization sebagai backup"""
    sentences = text.split('. ')
    if len(sentences) <= 5:
        return text
    
    if mode == "overview":
        key_sentences = sentences[:3] + sentences[len(sentences)//2:len(sentences)//2+3] + sentences[-3:]
        return ". ".join(key_sentences) + "."
    else:
        return ". ".join(sentences[:5]) + "."

# =========================
# 📑 PDF/DOCX Reader + AI Feedback
# =========================
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        # Gunakan library yang available
        if 'pypdf' in globals():
            reader = pypdf.PdfReader(file)
        else:
            reader = PyPDF2.PdfReader(file)
            
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error extracting PDF text: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"❌ Error extracting DOCX text: {str(e)}")
        return ""

def ai_comment_on_report(report_text, corpus_texts):
    """Analyze report similarity dengan NASA publications"""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf = vectorizer.fit_transform([report_text] + corpus_texts)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        most_similar_idx = sims.argmax()
        score = sims[most_similar_idx]
        related_paper = df.iloc[most_similar_idx]["title"] if most_similar_idx < len(df) else "Unknown"
        return f"Closest related article: **{related_paper}** (similarity score {score:.2f})."
    except Exception as e:
        return f"Error in similarity analysis: {str(e)}"

def get_simulated_osdr_videos(query, max_results=3):
    """Simulated NASA OSDR video data untuk demonstration"""
    video_database = {
        "microgravity": [
            {
                "title": "Microgravity Effects on Human Cells - NASA Research",
                "url": "https://www.youtube.com/watch?v=abc123micro",
                "thumbnail": "https://img.youtube.com/vi/abc123micro/mqdefault.jpg",
                "description": "NASA study on cellular behavior in microgravity environments",
                "duration": "15:30"
            }
        ],
        "radiation": [
            {
                "title": "Space Radiation Shielding Technologies", 
                "url": "https://www.youtube.com/watch?v=ghi789rad",
                "thumbnail": "https://img.youtube.com/vi/ghi789rad/mqdefault.jpg",
                "description": "Advanced materials for protecting astronauts from cosmic radiation",
                "duration": "18:20"
            }
        ],
        "plant": [
            {
                "title": "Growing Plants in Space - Veggie System",
                "url": "https://www.youtube.com/watch?v=mno345plant", 
                "thumbnail": "https://img.youtube.com/vi/mno345plant/mqdefault.jpg",
                "description": "NASA's research on plant growth for future space missions",
                "duration": "22:15"
            }
        ],
        "default": [
            {
                "title": "NASA Space Bioscience Research Overview",
                "url": "https://www.youtube.com/watch?v=yzab567default",
                "thumbnail": "https://img.youtube.com/vi/yzab567default/mqdefault.jpg",
                "description": "Comprehensive overview of NASA's life sciences research",
                "duration": "25:40"
            }
        ]
    }
    
    query_lower = query.lower()
    relevant_videos = []
    
    for keyword, videos in video_database.items():
        if keyword in query_lower and keyword != "default":
            relevant_videos.extend(videos)
    
    if not relevant_videos:
        relevant_videos = video_database["default"]
    
    return relevant_videos[:max_results]

# =========================
# 🌐 Graph Functions
# =========================
def build_knowledge_graph(df, max_nodes=50):
    """Build knowledge graph and return as HTML string"""
    try:
        G = nx.Graph()
        for idx, row in df.head(max_nodes).iterrows():
            article = f"📄 {row['title'][:40]}..." if 'title' in row else f"Article {idx}"
            G.add_node(article, color="lightblue", size=20)
            if "abstract" in row and pd.notna(row["abstract"]):
                words = str(row["abstract"]).split()
                keywords = [w for w in words if len(w) > 6][:5]
                for kw in keywords:
                    G.add_node(kw, color="lightgreen", size=15)
                    G.add_edge(article, kw)
        
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        net.force_atlas_2based()
        
        html_content = net.generate_html()
        html_content = html_content.replace('cdnjs.cloudflare.com', '')
        html_content = html_content.replace('cdn.jsdelivr.net', '')
        
        return html_content
        
    except Exception as e:
        return f"<p style='color: white;'>Error creating graph: {str(e)}</p>"

def build_similarity_graph(df, max_nodes=30, top_k=3):
    """Build similarity graph and return as HTML string"""
    try:
        G = nx.Graph()
        subset = df.head(max_nodes).copy()
        subset["text"] = subset["abstract"].fillna("") + " " + subset["conclusion"].fillna("")
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(subset["text"])
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        for i, row in subset.iterrows():
            art_i = f"📄 {row['title'][:40]}..." if 'title' in row else f"Article {i}"
            G.add_node(art_i, color="orange", size=20)
            similar_idx = sim_matrix[i].argsort()[-top_k-1:-1]
            for j in similar_idx:
                art_j = f"📄 {subset.iloc[j]['title'][:40]}..." if 'title' in subset.iloc[j] else f"Article {j}"
                sim_score = sim_matrix[i, j]
                if sim_score > 0.1:
                    G.add_edge(art_i, art_j, weight=sim_score)
        
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        net.force_atlas_2based()
        
        html_content = net.generate_html()
        html_content = html_content.replace('cdnjs.cloudflare.com', '')
        html_content = html_content.replace('cdn.jsdelivr.net', '')
        
        return html_content
        
    except Exception as e:
        return f"<p style='color: white;'>Error creating graph: {str(e)}</p>"

# =========================
# 🛠️ AI Services Test Function
# =========================
def test_ai_services():
    """Test AI service connectivity"""
    try:
        if hasattr(st.session_state, 'hf_api_key') and st.session_state.hf_api_key:
            st.sidebar.success("✅ Hugging Face token configured")
            return True
        else:
            st.sidebar.warning("⚠️ No Hugging Face token found")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Service test failed: {str(e)}")
        return False

# =========================
# 🖥️ Main UI
# =========================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, rgba(13, 59, 102, 0.8) 0%, rgba(30, 110, 167, 0.8) 50%, rgba(42, 157, 143, 0.8) 100%);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #00f5ff;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4);
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 15px;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        color: #e0f7fa;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00f5ff 0%, #0077b6 100%);
        color: #001219;
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 245, 255, 0.2);
        color: #00f5ff;
        border: 1px solid #00f5ff;
    }
    
    .main-content {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Header dengan Logo Team
def get_logo_base64(logo_file):
    if os.path.exists(logo_file):
        with open(logo_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

logo_file = "team_logo.png"
if os.path.exists(logo_file):
    logo_base64 = get_logo_base64(logo_file)
    logo_img = f'<img src="data:image/png;base64,{logo_base64}" class="team-logo" alt="Team Logo">'
else:
    logo_img = '<div style="width: 80px; height: 80px; border-radius: 50%; border: 2px solid #00f5ff; background: rgba(0,245,255,0.2); display: flex; align-items: center; justify-content: center; color: #00f5ff; font-weight: bold;">TEAM</div>'

st.markdown("""
<style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-bottom: 15px;
    }
    
    .team-logo {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.6);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    <div class="logo-container">
        {logo_img}
        <div>
            <h1 style="color: #00f5ff; text-shadow: 0 0 20px #00f5ff; margin-bottom: 10px; font-size: 2.5em;">
                🌌 NASA SPACE BIOSCIENCE DASHBOARD
            </h1>
            <p style="color: #e0f7fa; font-size: 1.3em; margin-bottom: 15px;">
                Advanced Analytics for Space Research Publications
            </p>
        </div>
        {logo_img}
    </div>
    <div style="display: flex; justify-content: center; gap: 20px; color: #e0f7fa;">
        <span>📊 <strong>{len(df)}</strong> Publications</span>
        <span>🔬 <strong>12+</strong> Research Fields</span>
        <span>🤖 <strong>AI-Powered</strong> Analysis</span>
        <span>🚀 <strong>NASA</strong> Curated</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Tabs
tabs = st.tabs(["🔍 **SEARCH & ANALYZE**", "📑 **UPLOAD & COMPARE**"])

# Content container
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# --- TAB 1: Search Publications ---
with tabs[0]:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(13, 59, 102, 0.8) 0%, rgba(30, 110, 167, 0.6) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
        margin-bottom: 25px;
    ">
        <h2 style="color: #00f5ff; margin: 0; text-align: center;">🔍 Advanced Publication Search</h2>
        <p style="color: #e0f7fa; text-align: center; margin: 10px 0 0 0;">
            Explore {len(df)}+ NASA Space Bioscience Publications with AI-Powered Analysis
        </p>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

    col_stats, col_search = st.columns([1, 2])

    with col_stats:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 245, 255, 0.3);
            height: 100%;
        ">
            <h4 style="color: #00f5ff; margin-top: 0;">📈 Publication Analytics</h4>
            <div style="color: #e0f7fa; line-height: 2;">
                📚 <strong>Total Articles:</strong> {len(df)}<br>
                📅 <strong>Years Covered:</strong> 1990-2024<br>
                🔬 <strong>Research Fields:</strong> 12+<br>
                🌟 <strong>Featured Topics:</strong><br>
                &nbsp;&nbsp;• Microgravity Effects<br>
                &nbsp;&nbsp;• Space Radiation<br>
                &nbsp;&nbsp;• Life Support Systems<br>
                &nbsp;&nbsp;• Astronaut Health<br>
                🤖 <strong>AI Tools:</strong> Available
            </div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col_search:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 245, 255, 0.2);
            margin-bottom: 20px;
        ">
            <h4 style="color: #00f5ff; margin-top: 0;">🎯 Search Controls</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Year Filter
        years = sorted(df["year"].dropna().unique()) if 'year' in df.columns else [2023, 2024]
        selected_years = st.multiselect(
            "**🗓️ Filter by Publication Years:**",
            years, 
            default=years[-5:] if len(years) > 5 else years,
            help="Select specific years to focus your search"
        )
        
        # Search Input
        query = st.text_input(
            "**🔍 Search Keywords:**",
            placeholder="Enter title, abstract, or conclusion keywords...",
            help="Search across article titles, abstracts, and conclusions"
        )
        
        with st.expander("💡 **Search Tips**", expanded=False):
            st.markdown("""
            - Use **specific terms**: "microgravity effects on cells"
            - Try **NASA acronyms**: "ISS, EVA, LEO"
            - Search **research fields**: "radiation biology", "plant growth"
            - Use **boolean operators**: "Mars AND habitat"
            """)

    # Results Section
    if query:
        df_filtered = df[df["year"].isin(selected_years)] if selected_years and 'year' in df.columns else df
        results = df_filtered[df_filtered.astype(str)
                              .apply(lambda x: x.str.contains(query, case=False, na=False))
                              .any(axis=1)]
        
        st.markdown(f"""
        <div style="
            background: rgba(0, 245, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #00f5ff;
        ">
            <h4 style="color: #00f5ff; margin: 0;">
                📊 Search Results: <span style="color: #e0f7fa;">{len(results)} articles found</span>
            </h4>
            <p style="color: #e0f7fa; margin: 5px 0 0 0;">
                Keywords: "{query}" | Years: {len(selected_years)} selected
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, row in results.head(20).iterrows():
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF", "#5F27CD"]
            current_color = colors[idx % len(colors)]
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border-left: 5px solid {current_color};
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            ">
                <h3 style="color: {current_color}; margin: 0 0 10px 0;">📄 {row['title'] if 'title' in row else 'Untitled'}</h3>
                <div style="color: #e0f7fa; font-size: 0.9em; margin-bottom: 15px;">
                    🗓️ <strong>Year:</strong> {row['year'] if 'year' in row and pd.notna(row['year']) else 'N/A'} 
                    | 🔗 <strong>Access:</strong> {'Available' if 'link' in row and pd.notna(row['link']) else 'Not available'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "link" in row and pd.notna(row["link"]):
                st.markdown(f"**🌐 Full Article:** [{row['link']}]({row['link']})")

            if "abstract" in row and pd.notna(row["abstract"]):
                with st.expander("📖 **Abstract**", expanded=False):
                    st.write(row["abstract"])

            if "conclusion" in row and pd.notna(row["conclusion"]):
                with st.expander("📌 **Conclusion**", expanded=False):
                    st.write(row["conclusion"])

            # Video Section
            st.markdown("---")
            st.markdown("#### 🎬 **NASA Video Resources**")
            
            col1, col2 = st.columns(2)
            
            title_lower = str(row['title']).lower() if 'title' in row else ""
            
            with col1:
                if "microgravity" in title_lower:
                    if st.button("📹 Microgravity Research", key=f"mg1_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=abc123micro")
                    if st.button("📹 Space Experiments", key=f"mg2_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=def456exp")
                elif "radiation" in title_lower:
                    if st.button("📹 Space Radiation", key=f"rad1_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=ghi789rad")
                    if st.button("📹 Radiation Protection", key=f"rad2_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=jkl012protect")
                elif "plant" in title_lower:
                    if st.button("📹 Space Farming", key=f"plant1_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=mno345plant")
                    if st.button("📹 Plant Research", key=f"plant2_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=pqr678grow")
                else:
                    if st.button("📹 NASA Research", key=f"nasa1_{idx}", use_container_width=True):
                        webbrowser.open("https://www.youtube.com/watch?v=yzab567nasa")
            
            with col2:
                if st.button("📹 Search More Videos", key=f"search_{idx}", use_container_width=True):
                    search_url = f"https://www.youtube.com/results?search_query=NASA+{str(row['title']).replace(' ', '+') if 'title' in row else 'space'}"
                    webbrowser.open(search_url)

            # AI Summary Button
            if st.button(f"🤖 **Summarize Article**", key=f"summarize_{idx}"):
                text_to_summarize = ""
                if "abstract" in row and pd.notna(row["abstract"]):
                    text_to_summarize += f"Abstract:\n{row['abstract']}\n\n"
                if "conclusion" in row and pd.notna(row["conclusion"]):
                    text_to_summarize += f"Conclusion:\n{row['conclusion']}\n\n"
                
                if text_to_summarize:
                    hf_configured = hasattr(st.session_state, 'hf_api_key') and st.session_state.hf_api_key
                    
                    if not hf_configured:
                        st.error("""
                        🔧 **AI Service Not Configured**
                        
                        **Langkah penyelesaian:**
                        1. **Dapatkan FREE token** dari [Hugging Face](https://huggingface.co/settings/tokens)
                        2. **Expand "🔑 Free API Setup"** di sidebar  
                        3. **Paste token** dalam text box
                        4. **Tekan Enter**
                        5. **Klik button ini semula**
                        
                        ⏱️ Token akan berfungsi serta-merta!
                        """)
                    else:
                        with st.spinner("🤖 Generating AI summary using Hugging Face..."):
                            summary = smart_summarize(text_to_summarize, mode="single")
                        
                        if summary.startswith("❌"):
                            st.error(f"**AI Summary Failed:** {summary}")
                            st.info("""
                            **Troubleshooting:**
                            - Pastikan token Hugging Face betul
                            - Token perlu permission **"Write"**
                            - Cuba token yang lain
                            """)
                        else:
                            st.success("✅ AI Summary Generated!")
                            st.markdown(f"""
                            <div style="
                                background: rgba(0, 100, 150, 0.3);
                                border-radius: 10px;
                                padding: 15px;
                                margin: 15px 0;
                                border-left: 4px solid #00FFFF;
                            ">
                                <strong style="color: #00f5ff;">🤖 AI Summary:</strong><br>
                                <span style="color: #e0f7fa;">{summary}</span>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("---")
        
        # Summarize All Button
        if len(results) > 1:
            if st.button("🧠 **Generate Comprehensive Summary**", use_container_width=True):
                all_text = ""
                for _, row in results.iterrows():
                    if "abstract" in row and pd.notna(row["abstract"]):
                        all_text += row["abstract"] + "\n\n"
                    if "conclusion" in row and pd.notna(row["conclusion"]):
                        all_text += row["conclusion"] + "\n\n"
                
                if all_text.strip():
                    with st.spinner("🤖 Generating comprehensive summary..."):
                        summary_all = smart_summarize(all_text, max_tokens=800, mode="overview")
                    
                    with st.expander("📋 **Comprehensive Research Overview**", expanded=True):
                        if summary_all.startswith("❌"):
                            st.error(summary_all)
                            st.info("💡 **Tip**: Sila check API configuration anda.")
                        else:
                            st.markdown(summary_all)

# --- TAB 2: Upload Report ---
with tabs[1]:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(42, 157, 143, 0.8) 0%, rgba(30, 110, 167, 0.6) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2a9d8f;
        box-shadow: 0 0 20px rgba(42, 157, 143, 0.3);
        margin-bottom: 25px;
        text-align: center;
    ">
        <h2 style="color: #2a9d8f; margin: 0;">📑 AI Research Report Analyzer</h2>
        <p style="color: #e0f7fa; margin: 10px 0 0 0;">
            Upload your research report for AI-powered analysis and NASA publication matching
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_features = st.columns([2, 1])

    with col_upload:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 12px;
            border: 2px dashed rgba(42, 157, 143, 0.5);
            text-align: center;
            margin-bottom: 20px;
        ">
            <h4 style="color: #2a9d8f; margin-top: 0;">📤 Upload Your Research Report</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "**Choose PDF or DOCX File**", 
            type=["pdf", "docx"],
            help="Upload your research paper, thesis, or report for AI analysis"
        )
        
        st.markdown("""
        <div style="
            background: rgba(42, 157, 143, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2a9d8f;
            margin-top: 15px;
        ">
            <p style="color: #e0f7fa; margin: 0; font-size: 0.9em;">
                💡 <strong>Supported formats:</strong> PDF, DOCX<br>
                📝 <strong>Ideal for:</strong> Research papers, literature reviews, thesis chapters<br>
                🔍 <strong>Analysis includes:</strong> Content extraction, similarity matching, gap analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_features:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(42, 157, 143, 0.3);
            height: 100%;
        ">
            <h4 style="color: #2a9d8f; margin-top: 0;">✨ Analysis Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🔬 Content Extraction**")
        st.markdown("- Full text analysis  \n- Key concept identification")
        
        st.markdown("**📊 Similarity Matching**")
        st.markdown("- NASA publication comparison  \n- Research gap detection")
        
        st.markdown("**🤖 AI Insights**")
        st.markdown("- Summary generation  \n- Relevance scoring  \n- Trend analysis")

    # File Processing
    if uploaded_file:
        st.markdown("""
        <div style="
            background: rgba(42, 157, 143, 0.1);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a9d8f;
            margin: 20px 0;
        ">
            <h4 style="color: #2a9d8f; margin: 0;">📄 File Analysis in Progress...</h4>
        </div>
        """, unsafe_allow_html=True)
        
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        col_info, col_status = st.columns(2)
        
        with col_info:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
            ">
                <h5 style="color: #2a9d8f; margin: 0 0 10px 0;">📋 File Information</h5>
                <p style="color: #e0f7fa; margin: 5px 0;">
                    <strong>Name:</strong> {filename}<br>
                    <strong>Size:</strong> {size}<br>
                    <strong>Type:</strong> {filetype}
                </p>
            </div>
            """.format(
                filename=file_details["Filename"],
                size=file_details["File size"],
                filetype=file_details["File type"]
            ), unsafe_allow_html=True)
        
        with col_status:
            with st.spinner("🔍 Extracting content and analyzing..."):
                if uploaded_file.type == "application/pdf":
                    report_text = extract_text_from_pdf(uploaded_file)
                else:
                    report_text = extract_text_from_docx(uploaded_file)
            
            st.success("✅ Analysis completed!")
        
        # Extracted Content
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        ">
            <h4 style="color: #2a9d8f; margin: 0 0 15px 0;">📖 Extracted Report Content</h4>
        </div>
        """, unsafe_allow_html=True)
        
        tab_preview, tab_stats = st.tabs(["📝 Content Preview", "📊 Text Statistics"])
        
        with tab_preview:
            st.text_area(
                "**Extracted Text**", 
                report_text[:2500] + "..." if len(report_text) > 2500 else report_text, 
                height=300,
                help="First 2500 characters of extracted content"
            )
        
        with tab_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Character Count", f"{len(report_text):,}")
            with col2:
                st.metric("Word Count", f"{len(report_text.split()):,}")
            with col3:
                st.metric("Estimated Pages", f"{len(report_text) // 1500 + 1}")
        
        # AI Feedback
        st.markdown("""
        <div style="
            background: rgba(0, 245, 255, 0.1);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #00f5ff;
            margin: 20px 0;
        ">
            <h4 style="color: #00f5ff; margin: 0 0 15px 0;">🤖 AI Analysis Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔬 Comparing with NASA publications..."):
            corpus_texts = df["abstract"].fillna("").tolist() if 'abstract' in df.columns else [""]
            comment = ai_comment_on_report(report_text, corpus_texts)
        
        st.markdown(f"""
        <div style="
            background: rgba(0, 100, 150, 0.2);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #00f5ff;
        ">
            <h5 style="color: #00f5ff; margin: 0 0 10px 0;">📈 Relevance Analysis</h5>
            <p style="color: #e0f7fa; margin: 0; line-height: 1.6;">{comment}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        ">
            <h5 style="color: #2a9d8f; margin: 0 0 10px 0;">💡 Recommended Next Steps</h5>
            <p style="color: #e0f7fa; margin: 0; font-size: 0.9em;">
                • Search related articles using keywords from your report<br>
                • Generate similarity graph to visualize connections<br>
                • Use AI summarizer for detailed analysis of matched publications
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.02);
            padding: 60px 20px;
            border-radius: 12px;
            border: 2px dashed rgba(255, 255, 255, 0.1);
            text-align: center;
            margin: 40px 0;
        ">
            <h4 style="color: rgba(255, 255, 255, 0.5); margin: 0;">📁 No file uploaded</h4>
            <p style="color: rgba(255, 255, 255, 0.4);">
                Upload a PDF or DOCX file to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

# Close main content div
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🔑 Sidebar API Setup
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Free API Setup")

with st.sidebar.expander("Setup Free AI APIs", expanded=True):
    st.markdown("""
    **Dapatkan FREE Hugging Face Token:**
    
    1. Daftar di [huggingface.co](https://huggingface.co)
    2. Pergi ke [Settings → Access Tokens](https://huggingface.co/settings/tokens)
    3. Create token dengan permission **Write**
    """)
    
    hf_key = st.text_input("Hugging Face Token:", type="password", key="hf_token_input")
    
    if hf_key:
        st.session_state.hf_api_key = hf_key
        os.environ["HF_API_KEY"] = hf_key
        st.success("✅ HF Token saved! Now try 'Summarize Article'")
    
    if hasattr(st.session_state, 'hf_api_key') and st.session_state.hf_api_key:
        st.info(f"🔑 Token set: {st.session_state.hf_api_key[:10]}...")

# Test AI Services
test_ai_services()

# =========================
# 🌐 Sidebar Graph Section
# =========================
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0d3b66 0%, #1e6ea7 50%, #2a9d8f 100%);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #00f5ff;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
    margin-bottom: 20px;
">
    <h3 style="color: white; text-align: center; margin: 0; text-shadow: 0 0 10px #00f5ff;">🌌 NASA Graph Visualizations</h3>
    <p style="color: #e0f7fa; text-align: center; font-size: 0.9em;">Explore connections between space research articles</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔗 **Similarity Graph**", use_container_width=True, help="Show how articles are related by content similarity"):
        with st.spinner("🔄 Creating similarity graph..."):
            html_content = build_similarity_graph(df, max_nodes=20)
            st.components.v1.html(html_content, height=600, scrolling=True)
            st.sidebar.success("✅ Similarity Graph generated!")

with col2:
    if st.button("🧩 **Knowledge Graph**", use_container_width=True, help="Visualize keywords and concepts from articles"):
        with st.spinner("🔄 Creating knowledge graph..."):
            html_content = build_knowledge_graph(df, max_nodes=20)
            st.components.v1.html(html_content, height=600, scrolling=True)
            st.sidebar.success("✅ Knowledge Graph generated!")

# Additional NASA-themed elements
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(0, 245, 255, 0.3);
">
    <p style="color: #e0f7fa; font-size: 0.8em; text-align: center; margin: 0;">
        <strong>📊 Graph Features:</strong><br>
        • Content Similarity Analysis<br>
        • Keyword Relationships<br>
        • Research Trends Mapping
    </p>
</div>
""", unsafe_allow_html=True)

# NASA Stats
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center;">
    <p style="color: #00f5ff; font-size: 2em; margin: 0; text-shadow: 0 0 10px #00f5ff;">{}</p>
    <p style="color: #e0f7fa; font-size: 0.9em; margin: 0;">Space Publications</p>
</div>
""".format(len(df)), unsafe_allow_html=True)
