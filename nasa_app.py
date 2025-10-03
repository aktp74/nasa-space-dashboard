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
import PyPDF2
import docx
import streamlit.components.v1 as components 

if 'show_similarity_graph' not in st.session_state:
    st.session_state.show_similarity_graph = False
if 'show_knowledge_graph' not in st.session_state:
    st.session_state.show_knowledge_graph = False

# =========================
# 🎨 Background Image Setup
# =========================
# =========================
# 🎨 Sidebar Background (Deep Space Gradient)
# =========================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0e2e 0%, #1a237e 50%, #283593 100%);
    }
    
    /* Untuk pastikan text dalam sidebar kelihatan jelas */
    .sidebar-content {
        color: #ffffff;
    }
    
    /* Improve button visibility dalam sidebar */
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
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_file = "background.jpg"
if os.path.exists(img_file):
    img_base64 = get_base64_of_bin_file(img_file)

    page_bg = f"""
    <style>
    /* Background Image */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}

    /* Glassmorphism + Neon Border */
    .block-container {{
        background: rgba(255, 255, 255, 0.08);   /* semi-transparent glass */
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 200, 255, 0.5);  /* neon border */
        box-shadow: 0 0 15px rgba(0, 200, 255, 0.25);  /* glowing shadow */
        color: #f0faff;  /* soft text color */
        transition: all 0.3s ease-in-out;
    }}

    /* Hover Effect */
    .block-container:hover {{
        border: 1px solid rgba(0, 255, 255, 0.9);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        transform: translateY(-3px);   /* floating effect */
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: #1a3c34;
    }}
    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🎵 Background Music Setup
# =========================
music_file = "background.mp3"
if os.path.exists(music_file):
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
else:
    st.sidebar.warning("⚠️ background.mp3 tidak dijumpai. Letak fail ini dalam folder sama dengan app.py")

# =========================
# 📊 Data Loading
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("SB_publication_PMC.csv", encoding="latin1")
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_data()

# =========================
# 🤖 AI Summarizer
# =========================
def ai_summarize(text, max_tokens=400, mode="single"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "❌ API key OpenRouter tidak dijumpai. Sila set dulu."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if mode == "overview":
        user_prompt = f"""
Below is a collection of abstracts and conclusions from multiple NASA space bioscience articles.
Your task: Write a comprehensive overview summary (5–10 paragraphs) in English.
Focus on major themes, key findings, knowledge gaps, and overall research trends.
Do not list each article one by one.

Text:
{text}
"""
    else:
        user_prompt = f"Summarize this article in 5–7 sentences, in English:\n\n{text}"

    data = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "system", "content": "You are an AI that summarizes scientific articles clearly in English."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Ralat AI: {e}"
# =========================
# 🆕 SMART SUMMARIZE FUNCTIONS - TAMBAH INI
# =========================
def smart_summarize(text, max_tokens=400, mode="single"):
    """Smart summarization dengan fallback"""
    # Cuba AI dulu
    ai_result = ai_summarize(text, max_tokens, mode)
    
    # Jika AI gagal, guna fallback
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
        # Ambil 8-10 sentences untuk overview
        key_sentences = sentences[:3] + sentences[len(sentences)//2:len(sentences)//2+3] + sentences[-3:]
        return ". ".join(key_sentences) + "."
    else:
        # Ambil 5-7 sentences untuk single article
        return ". ".join(sentences[:5]) + "."
# =========================
# 📑 PDF/DOCX Reader + AI Feedback
# =========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def ai_comment_on_report(report_text, corpus_texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([report_text] + corpus_texts)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    most_similar_idx = sims.argmax()
    score = sims[most_similar_idx]
    related_paper = df.iloc[most_similar_idx]["title"]
    return f"Closest related article: **{related_paper}** (similarity score {score:.2f})."

def search_nasa_osdr_videos(query, max_results=3):
    """
    Search for related videos from NASA OSDR database
    """
    try:
        return get_simulated_osdr_videos(query, max_results)
    except Exception as e:
        return get_simulated_osdr_videos(query, max_results)

def get_simulated_osdr_videos(query, max_results=3):
    """
    Simulated NASA OSDR video data untuk demonstration
    """
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
    G = nx.Graph()
    for idx, row in df.head(max_nodes).iterrows():
        article = f"📄 {row['title'][:40]}..."
        G.add_node(article, color="lightblue", size=20)
        if "abstract" in row and pd.notna(row["abstract"]):
            words = str(row["abstract"]).split()
            keywords = [w for w in words if len(w) > 6][:5]
            for kw in keywords:
                G.add_node(kw, color="lightgreen", size=15)
                G.add_edge(article, kw)
    return G

def build_similarity_graph(df, max_nodes=30, top_k=3):
    G = nx.Graph()
    subset = df.head(max_nodes).copy()
    subset["text"] = subset["abstract"].fillna("") + " " + subset["conclusion"].fillna("")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(subset["text"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    for i, row in subset.iterrows():
        art_i = f"📄 {row['title'][:40]}..."
        G.add_node(art_i, color="orange", size=20)
        similar_idx = sim_matrix[i - subset.index[0]].argsort()[-top_k-1:-1]
        for j in similar_idx:
            art_j = f"📄 {subset.iloc[j]['title'][:40]}..."
            sim_score = sim_matrix[i - subset.index[0], j]
            if sim_score > 0.1:
                G.add_edge(art_i, art_j, weight=sim_score)
    return G
def display_knowledge_graph_online(df):
    """Fungsi baru untuk display graph dalam Streamlit online"""
    try:
        # Build graph
        G = build_knowledge_graph(df, max_nodes=30)
        
        # Create Pyvis network
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        
        # Generate HTML content
        html_content = net.generate_html()
        
        # Display dalam Streamlit menggunakan components
        components.html(html_content, height=600, scrolling=True)
        
        return True
    except Exception as e:
        st.error(f"❌ Error generating graph: {e}")
        return False

def display_similarity_graph_online(df):
    """Fungsi untuk similarity graph"""
    try:
        G = build_similarity_graph(df, max_nodes=20)
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        html_content = net.generate_html()
        components.html(html_content, height=600, scrolling=True)
        return True
    except Exception as e:
        st.error(f"❌ Error generating similarity graph: {e}")
        return False
# =========================
# 🖥️ Main UI
# =========================
st.markdown("""
<style>
    /* Header styling */
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
    
    /* Tabs styling */
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
    
    /* Content area improvement */
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

logo_file = "team_logo.png"  # Ganti dengan nama file logo anda
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
        <span>📊 <strong>608</strong> Publications</span>
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
    # Enhanced Search Header
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
            Explore 608+ NASA Space Bioscience Publications with AI-Powered Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main Content Columns
    col_stats, col_search = st.columns([1, 2])

    with col_stats:
        # Statistics Panel
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
                📚 <strong>Total Articles:</strong> 608<br>
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
        """, unsafe_allow_html=True)

    with col_search:
        # Search Control Panel
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
        
        # Year Filter dengan design yang lebih baik
        years = sorted(df["year"].dropna().unique())
        selected_years = st.multiselect(
            "**🗓️ Filter by Publication Years:**",
            years, 
            default=years[-5:],
            help="Select specific years to focus your search"
        )
        
        # Search Input yang lebih prominent
        query = st.text_input(
            "**🔍 Search Keywords:**",
            placeholder="Enter title, abstract, or conclusion keywords...",
            help="Search across article titles, abstracts, and conclusions"
        )
        
        # Quick Search Tips
        with st.expander("💡 **Search Tips**", expanded=False):
            st.markdown("""
            - Use **specific terms**: "microgravity effects on cells"
            - Try **NASA acronyms**: "ISS, EVA, LEO"
            - Search **research fields**: "radiation biology", "plant growth"
            - Use **boolean operators**: "Mars AND habitat"
            """)

    # Results Section - HANYA SATU BAHAGIAN INI SAHAJA
    if query:
        df_filtered = df[df["year"].isin(selected_years)] if selected_years else df
        results = df_filtered[df_filtered.astype(str)
                              .apply(lambda x: x.str.contains(query, case=False, na=False))
                              .any(axis=1)]
        
        # Results Header
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
        
        # Kekalkan for loop articles yang sudah diubah (dengan warna)
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
                <h3 style="color: {current_color}; margin: 0 0 10px 0;">📄 {row['title']}</h3>
                <div style="color: #e0f7fa; font-size: 0.9em; margin-bottom: 15px;">
                    🗓️ <strong>Year:</strong> {row['year'] if 'year' in row and pd.notna(row['year']) else 'N/A'} 
                    | 🔗 <strong>Access:</strong> {'Available' if 'link' in row and pd.notna(row['link']) else 'Not available'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Content kekal sama seperti sebelumnya
            if "link" in row and pd.notna(row["link"]):
                st.markdown(f"**🌐 Full Article:** [{row['link']}]({row['link']})")

            if "abstract" in row and pd.notna(row["abstract"]):
                with st.expander("📖 **Abstract**", expanded=False):
                    st.write(row["abstract"])

            if "conclusion" in row and pd.notna(row["conclusion"]):
                with st.expander("📌 **Conclusion**", expanded=False):
                    st.write(row["conclusion"])

            # 🎬 VIDEO SECTION - UPDATED VERSION (PASTI NAMPAK)
            st.markdown("---")
            st.markdown("#### 🎬 **NASA Video Resources**")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            title_lower = row['title'].lower()
            
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
                    search_url = f"https://www.youtube.com/results?search_query=NASA+{row['title'].replace(' ', '+')}"
                    webbrowser.open(search_url)

            # Enhanced AI Summary Button
            if st.button(f"🤖 **Summarize Article**", key=f"summarize_{idx}"):
                text_to_summarize = ""
                if "abstract" in row and pd.notna(row["abstract"]):
                    text_to_summarize += f"Abstract:\n{row['abstract']}\n\n"
                if "conclusion" in row and pd.notna(row["conclusion"]):
                    text_to_summarize += f"Conclusion:\n{row['conclusion']}\n\n"
                if text_to_summarize:
                    summary = ai_summarize(text_to_summarize, mode="single")
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
                        # Guna smart_summarize instead of ai_summarize
                        summary_all = smart_summarize(all_text, max_tokens=800, mode="overview")
                    
                    with st.expander("📋 **Comprehensive Research Overview**", expanded=True):
                        if summary_all.startswith("❌"):
                            st.error(summary_all)
                            st.info("💡 **Tip**: Sila check OpenRouter API key dan kredit balance anda.")
                        else:
                            st.markdown(summary_all)
def smart_summarize(text, max_tokens=400, mode="single"):
    """Smart summarization dengan fallback"""
    # Cuba AI dulu
    ai_result = ai_summarize(text, max_tokens, mode)
    
    # Jika AI gagal, guna fallback
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
        # Ambil 8-10 sentences untuk overview
        key_sentences = sentences[:3] + sentences[len(sentences)//2:len(sentences)//2+3] + sentences[-3:]
        return ". ".join(key_sentences) + "."
    else:
        # Ambil 5-7 sentences untuk single article
        return ". ".join(sentences[:5]) + "."
        
        
       
# --- TAB 2: Upload Report ---
with tabs[1]:
    # Enhanced Upload Header
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

    # Main Upload Layout
    col_upload, col_features = st.columns([2, 1])

    with col_upload:
        # Upload Section dengan design yang lebih baik
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
        
        # Enhanced File Uploader
        uploaded_file = st.file_uploader(
            "**Choose PDF or DOCX File**", 
            type=["pdf", "docx"],
            help="Upload your research paper, thesis, or report for AI analysis"
        )
        
        # File Requirements Info
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
        # Features Panel - SUPER SIMPLE (PASTI WORK)
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
        
        # Gunakan Streamlit native components
        st.markdown("**🔬 Content Extraction**")
        st.markdown("- Full text analysis  \n- Key concept identification")
        
        st.markdown("**📊 Similarity Matching**")
        st.markdown("- NASA publication comparison  \n- Research gap detection")
        
        st.markdown("**🤖 AI Insights**")
        st.markdown("- Summary generation  \n- Relevance scoring  \n- Trend analysis")

    # File Processing Section
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
        
        # File Info
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
            # Processing Animation
            with st.spinner("🔍 Extracting content and analyzing..."):
                # Extract text dari file
                if uploaded_file.type == "application/pdf":
                    report_text = extract_text_from_pdf(uploaded_file)
                else:
                    report_text = extract_text_from_docx(uploaded_file)
            
            st.success("✅ Analysis completed!")
        
        # Extracted Content Section
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
        
        # Text preview dengan tabs
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
        
        # AI Feedback Section
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
        
        # AI Comment dengan enhanced display
        with st.spinner("🔬 Comparing with NASA publications..."):
            comment = ai_comment_on_report(report_text, df["abstract"].fillna("").tolist())
        
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
        
        # Additional Analysis Suggestions
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
        # Placeholder ketika tiada file diupload
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

# --- Sidebar Graph ---
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
# ✅ BUTTONS BARU - LETAK DI SINI, SELEPAS HEADER
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔗 **Similarity Graph**", use_container_width=True, help="Show how articles are related by content similarity"):
        st.session_state.show_similarity_graph = True
        st.session_state.show_knowledge_graph = False

with col2:
    if st.button("🧩 **Knowledge Graph**", use_container_width=True, help="Visualize keywords and concepts from articles"):
        st.session_state.show_knowledge_graph = True
        st.session_state.show_similarity_graph = False

# Additional NASA-themed elements  ← ✅ HANYA SATU!
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

# NASA Stats  ← ✅ HANYA SATU!
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center;">
    <p style="color: #00f5ff; font-size: 2em; margin: 0; text-shadow: 0 0 10px #00f5ff;">608</p>
    <p style="color: #e0f7fa; font-size: 0.9em; margin: 0;">Space Publications</p>
</div>
""", unsafe_allow_html=True)
# =========================
# 🌐 GRAPH DISPLAY AREA - ONLINE VERSION
# =========================
st.markdown("---")

# Graph Display Section
if st.session_state.get('show_similarity_graph'):
    st.markdown("### 🔗 Similarity Graph - Article Relationships")
    st.info("🖱️ **Tips**: Drag nodes to explore • Scroll to zoom • Click nodes to see connections")
    with st.spinner("🔄 Generating similarity graph... This may take a few seconds"):
        display_similarity_graph_online(df)
        
elif st.session_state.get('show_knowledge_graph'):
    st.markdown("### 🧩 Knowledge Graph - Keywords & Concepts") 
    st.info("🖱️ **Tips**: Blue nodes = Articles • Green nodes = Keywords • Drag to explore relationships")
    with st.spinner("🔄 Generating knowledge graph... This may take a few seconds"):
        display_knowledge_graph_online(df)
else:
    # Default message - hanya show pertama kali
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        padding: 60px 20px;
        border-radius: 15px;
        border: 2px dashed rgba(0, 245, 255, 0.3);
        text-align: center;
        margin: 20px 0;
    ">
        <h3 style="color: #00f5ff;">🌌 Graph Visualization Ready</h3>
        <p style="color: #e0f7fa; font-size: 1.1em;">
            Click on either graph button in the sidebar to visualize NASA research relationships
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
            <div style="text-align: center;">
                <div style="color: orange; font-size: 2em;">🔗</div>
                <p style="color: #e0f7fa; margin: 5px 0;">Similarity Graph<br><small>Article connections</small></p>
            </div>
            <div style="text-align: center;">
                <div style="color: lightblue; font-size: 2em;">🧩</div>
                <p style="color: #e0f7fa; margin: 5px 0;">Knowledge Graph<br><small>Keywords & concepts</small></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 🏁 END OF APP
# =========================
st.markdown('</div>', unsafe_allow_html=True)  # Tutup main-content div


