import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CODE-BASED BACKGROUND ILLUSTRATION
# Laptop + Mouse + Mobile + Books + Coffee + Desk
# No external image required
# ============================================================

svg_background = """
<svg xmlns="http://www.w3.org/2000/svg"
     width="1600"
     height="1000"
     viewBox="0 0 1600 1000">

<defs>

    <linearGradient id="desk" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#111827"/>
        <stop offset="45%" stop-color="#172554"/>
        <stop offset="100%" stop-color="#020617"/>
    </linearGradient>

    <linearGradient id="screen" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#172554"/>
        <stop offset="50%" stop-color="#312e81"/>
        <stop offset="100%" stop-color="#020617"/>
    </linearGradient>

    <linearGradient id="purpleGlow">
        <stop offset="0%" stop-color="#8b5cf6"/>
        <stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>

    <radialGradient id="light">
        <stop offset="0%" stop-color="#6366f1" stop-opacity=".55"/>
        <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
    </radialGradient>

    <filter id="blur">
        <feGaussianBlur stdDeviation="45"/>
    </filter>

    <filter id="soft">
        <feGaussianBlur stdDeviation="12"/>
    </filter>

</defs>


<!-- BACKGROUND -->

<rect width="1600" height="1000" fill="#030712"/>

<circle cx="200" cy="160" r="300"
        fill="url(#light)"
        filter="url(#blur)"/>

<circle cx="1350" cy="230" r="350"
        fill="#7c3aed"
        opacity=".18"
        filter="url(#blur)"/>

<circle cx="800" cy="850" r="350"
        fill="#2563eb"
        opacity=".15"
        filter="url(#blur)"/>


<!-- WALL -->

<rect x="0" y="0"
      width="1600"
      height="630"
      fill="#071126"/>


<!-- DESK -->

<path d="M0 610 L1600 520 L1600 1000 L0 1000 Z"
      fill="url(#desk)"/>


<!-- DESK LIGHT -->

<ellipse cx="760"
         cy="720"
         rx="600"
         ry="180"
         fill="#4f46e5"
         opacity=".10"
         filter="url(#soft)"/>


<!-- LAPTOP SCREEN -->

<g transform="translate(810 80)">

    <!-- screen body -->

    <rect x="0"
          y="0"
          width="500"
          height="330"
          rx="22"
          fill="#111827"
          stroke="#475569"
          stroke-width="6"/>

    <!-- display -->

    <rect x="22"
          y="22"
          width="456"
          height="270"
          rx="10"
          fill="url(#screen)"/>

    <!-- code lines -->

    <g opacity=".85">

        <rect x="55" y="65"
              width="230" height="8"
              rx="4"
              fill="#60a5fa"/>

        <rect x="55" y="95"
              width="320" height="7"
              rx="4"
              fill="#a78bfa"/>

        <rect x="55" y="125"
              width="270" height="7"
              rx="4"
              fill="#38bdf8"/>

        <rect x="55" y="155"
              width="350" height="7"
              rx="4"
              fill="#818cf8"/>

        <rect x="55" y="185"
              width="220" height="7"
              rx="4"
              fill="#c084fc"/>

        <rect x="55" y="215"
              width="300" height="7"
              rx="4"
              fill="#60a5fa"/>

    </g>

    <!-- screen glow -->

    <circle cx="380"
            cy="80"
            r="70"
            fill="#6366f1"
            opacity=".15"
            filter="url(#soft)"/>

    <!-- laptop base -->

    <path d="M-55 335
             L555 335
             L615 390
             L-115 390 Z"
          fill="#334155"
          stroke="#64748b"
          stroke-width="4"/>

    <!-- trackpad -->

    <rect x="220"
          y="345"
          width="160"
          height="30"
          rx="8"
          fill="#1e293b"/>

</g>


<!-- BOOK STACK -->

<g transform="translate(1320 500)">

    <rect x="0"
          y="0"
          width="190"
          height="55"
          rx="8"
          fill="#312e81"/>

    <rect x="15"
          y="-65"
          width="180"
          height="55"
          rx="8"
          fill="#1d4ed8"/>

    <rect x="0"
          y="-130"
          width="200"
          height="55"
          rx="8"
          fill="#7c3aed"/>

    <text x="25"
          y="-95"
          fill="#ffffff"
          font-size="22"
          font-family="Arial">
        MACHINE LEARNING
    </text>

    <text x="30"
          y="-30"
          fill="#ffffff"
          font-size="22"
          font-family="Arial">
        PYTHON
    </text>

    <text x="35"
          y="35"
          fill="#ffffff"
          font-size="22"
          font-family="Arial">
        DATA SCIENCE
    </text>

</g>


<!-- MOUSE -->

<g transform="translate(1190 690)">

    <ellipse cx="100"
             cy="80"
             rx="90"
             ry="125"
             fill="#1e293b"
             stroke="#64748b"
             stroke-width="4"/>

    <path d="M100 0 L100 95"
          stroke="#64748b"
          stroke-width="4"/>

    <circle cx="100"
            cy="42"
            r="9"
            fill="#8b5cf6"/>

    <ellipse cx="65"
             cy="55"
             rx="20"
             ry="35"
             fill="#6366f1"
             opacity=".12"/>

</g>


<!-- MOBILE PHONE -->

<g transform="translate(1380 720) rotate(-12)">

    <rect x="0"
          y="0"
          width="115"
          height="210"
          rx="18"
          fill="#111827"
          stroke="#64748b"
          stroke-width="5"/>

    <rect x="12"
          y="30"
          width="91"
          height="145"
          rx="8"
          fill="#172554"/>

    <circle cx="58"
            cy="193"
            r="7"
            fill="#475569"/>

    <circle cx="58"
            cy="18"
            r="5"
            fill="#475569"/>

</g>


<!-- COFFEE CUP -->

<g transform="translate(610 590)">

    <ellipse cx="80"
             cy="145"
             rx="75"
             ry="20"
             fill="#020617"
             opacity=".45"/>

    <path d="M20 40
             L145 40
             L130 140
             Q80 175 35 140 Z"
          fill="#f8fafc"/>

    <path d="M145 65
             Q195 60 175 105
             Q160 130 132 120"
          fill="none"
          stroke="#f8fafc"
          stroke-width="15"/>

    <ellipse cx="82"
             cy="42"
             rx="62"
             ry="16"
             fill="#451a03"/>

    <path d="M60 0 Q40 -40 65 -70"
          stroke="#cbd5e1"
          stroke-width="5"
          fill="none"
          opacity=".4"/>

</g>


<!-- NOTEBOOK -->

<g transform="translate(360 760) rotate(-8)">

    <rect x="0"
          y="0"
          width="300"
          height="190"
          rx="12"
          fill="#1e3a8a"
          stroke="#818cf8"
          stroke-width="3"/>

    <line x1="35" y1="50"
          x2="260" y2="50"
          stroke="#93c5fd"
          stroke-width="4"/>

    <line x1="35" y1="85"
          x2="230" y2="85"
          stroke="#93c5fd"
          stroke-width="4"/>

    <line x1="35" y1="120"
          x2="250" y2="120"
          stroke="#93c5fd"
          stroke-width="4"/>

    <line x1="35" y1="155"
          x2="190" y2="155"
          stroke="#93c5fd"
          stroke-width="4"/>

</g>


<!-- SMALL DOCUMENT ICON -->

<g transform="translate(90 600)">

    <rect width="150"
          height="190"
          rx="12"
          fill="#111827"
          stroke="#6366f1"
          stroke-width="3"/>

    <path d="M35 45
             L115 45
             M35 80
             L115 80
             M35 115
             L100 115
             M35 150
             L90 150"
          stroke="#818cf8"
          stroke-width="8"
          stroke-linecap="round"/>

</g>


<!-- DECORATIVE GLOW -->

<circle cx="1050"
        cy="500"
        r="180"
        fill="#6366f1"
        opacity=".07"
        filter="url(#soft)"/>

</svg>
"""

svg_base64 = base64.b64encode(
    svg_background.encode()
).decode()


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
f"""
<style>

/* =========================================================
   CODE GENERATED TECH-OFFICE BACKGROUND
   ========================================================= */

.stApp {{

    background-color: #030712;

    background-image:
        linear-gradient(
            rgba(3,7,18,0.72),
            rgba(3,7,18,0.82)
        ),
        url("data:image/svg+xml;base64,{svg_base64}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}


/* =========================================================
   CONTENT
   ========================================================= */

.block-container {{

    max-width: 1420px;

    padding-top: 2rem;
    padding-bottom: 4rem;

    position: relative;
    z-index: 5;
}}


/* =========================================================
   TEXT
   ========================================================= */

.stApp,
.stApp p,
.stApp span,
.stApp label {{

    color: #e5e7eb;
}}


h1 {{

    color: white !important;

    font-size: 48px !important;

    font-weight: 900 !important;

    line-height: 1.05 !important;

    text-shadow:
        0 0 25px rgba(96,165,250,0.45);
}}


h2,
h3 {{

    color: white !important;

    font-weight: 800 !important;
}}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {{

    background:
        linear-gradient(
            180deg,
            rgba(2,6,23,0.97),
            rgba(15,23,42,0.95),
            rgba(30,27,75,0.96)
        );

    border-right:
        1px solid rgba(99,102,241,0.28);

    box-shadow:
        10px 0 40px rgba(0,0,0,0.25);
}}


section[data-testid="stSidebar"] * {{

    color: #e2e8f0 !important;
}}


/* =========================================================
   GLASS CARDS
   ========================================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {{

    background:
        linear-gradient(
            135deg,
            rgba(20,31,72,0.78),
            rgba(7,14,38,0.68)
        );

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    border:
        1px solid rgba(129,140,248,0.30) !important;

    border-radius:
        24px !important;

    box-shadow:
        0 20px 60px rgba(0,0,0,0.38),
        inset 0 1px 0 rgba(255,255,255,0.08);

    transition:
        all .35s ease;

    animation:
        cardEnter .65s ease-out;
}}


div[data-testid="stVerticalBlockBorderWrapper"]:hover {{

    transform:
        translateY(-5px);

    border-color:
        rgba(129,140,248,0.65) !important;

    box-shadow:
        0 30px 70px rgba(0,0,0,0.45),
        0 0 35px rgba(79,70,229,0.12);
}}


/* =========================================================
   TEXT AREA
   ========================================================= */

textarea {{

    background:
        rgba(3,10,30,0.78) !important;

    color:
        #f1f5f9 !important;

    border:
        1px solid rgba(96,165,250,0.35) !important;

    border-radius:
        17px !important;

    transition:
        .3s ease !important;
}}


textarea::placeholder {{

    color:
        #7c8aaa !important;
}}


textarea:focus {{

    border-color:
        #818cf8 !important;

    box-shadow:
        0 0 0 4px rgba(99,102,241,.12),
        0 0 30px rgba(79,70,229,.12) !important;
}}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {{

    background:
        linear-gradient(
            135deg,
            rgba(30,41,90,.65),
            rgba(8,15,38,.75)
        );

    border:
        2px dashed rgba(129,140,248,.48);

    border-radius:
        20px;

    padding:
        14px;

    transition:
        .35s ease;
}}


[data-testid="stFileUploader"]:hover {{

    border-color:
        #8b5cf6;

    box-shadow:
        0 0 30px rgba(124,58,237,.16);

    transform:
        translateY(-3px);
}}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {{

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed,
            #2563eb
        ) !important;

    color:
        white !important;

    border:
        none !important;

    border-radius:
        16px !important;

    min-height:
        54px;

    font-size:
        17px !important;

    font-weight:
        850 !important;

    box-shadow:
        0 12px 35px rgba(79,70,229,.40);

    transition:
        all .3s ease !important;
}}


.stButton > button:hover {{

    transform:
        translateY(-5px)
        scale(1.01);

    box-shadow:
        0 20px 50px rgba(79,70,229,.55);
}}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {{

    background:
        linear-gradient(
            135deg,
            rgba(30,41,90,.75),
            rgba(15,23,55,.70)
        );

    border:
        1px solid rgba(129,140,248,.25);

    border-radius:
        20px;

    padding:
        18px;

    box-shadow:
        0 15px 35px rgba(0,0,0,.25);

    transition:
        .3s ease;
}}


div[data-testid="stMetric"]:hover {{

    transform:
        translateY(-6px);

    border-color:
        rgba(129,140,248,.65);

    box-shadow:
        0 20px 45px rgba(79,70,229,.20);
}}


div[data-testid="stMetricLabel"] {{

    color:
        #94a3b8 !important;
}}


div[data-testid="stMetricValue"] {{

    color:
        #a5b4fc !important;

    font-weight:
        900 !important;
}}


/* =========================================================
   PROGRESS
   ========================================================= */

div[data-testid="stProgressBar"] {{

    background:
        rgba(15,23,42,.8);

    border-radius:
        20px;
}}


div[data-testid="stProgressBar"] > div > div {{

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #8b5cf6,
            #06b6d4
        ) !important;

    border-radius:
        20px;
}}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {{

    border-radius:
        16px;

    background:
        rgba(15,23,42,.75);

    backdrop-filter:
        blur(12px);
}}


/* =========================================================
   ANIMATIONS
   ========================================================= */

@keyframes cardEnter {{

    from {{
        opacity: 0;
        transform: translateY(18px);
    }}

    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}


@keyframes glowPulse {{

    0%,100% {{
        box-shadow:
            0 0 20px rgba(99,102,241,.15);
    }}

    50% {{
        box-shadow:
            0 0 45px rgba(99,102,241,.28);
    }}
}}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {{

    h1 {{
        font-size: 31px !important;
    }}

    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def extract_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def calculate_match_score(
    job_description,
    resume_text
):

    documents = [
        clean_text(job_description),
        clean_text(resume_text)
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(
            similarity * 100,
            2
        )

    except Exception:

        return 0.0


# ============================================================
# SKILLS
# ============================================================

SKILLS = [

    "python",
    "java",
    "c++",
    "c",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",

    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",

    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "keras",

    "power bi",
    "tableau",
    "excel",

    "html",
    "css",
    "javascript",
    "react",
    "node.js",

    "django",
    "flask",
    "streamlit",

    "aws",
    "azure",

    "git",
    "github",

    "natural language processing",
    "nlp"
]


def extract_skills(text):

    text_lower = text.lower()

    return [
        skill
        for skill in SKILLS
        if skill in text_lower
    ]


def extract_jd_skills(job_description):

    text_lower = job_description.lower()

    return [
        skill
        for skill in SKILLS
        if skill in text_lower
    ]


# ============================================================
# EDUCATION
# ============================================================

def extract_education(text):

    lines = text.split("\n")

    keywords = [

        "education",
        "b.tech",
        "btech",
        "b.e",
        "b.e.",
        "b.sc",
        "bsc",

        "m.tech",
        "mtech",
        "m.sc",
        "msc",

        "mba",
        "bca",
        "mca",

        "bachelor",
        "master",

        "university",
        "college",
        "degree"
    ]

    result = []

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        if any(
            keyword in clean.lower()
            for keyword in keywords
        ):

            result.append(clean)

    return result[:8]


# ============================================================
# EXPERIENCE
# ============================================================

def extract_experience(text):

    lines = text.split("\n")

    keywords = [

        "experience",
        "intern",
        "internship",

        "developer",
        "engineer",
        "analyst",
        "manager",

        "software",
        "worked",

        "employment",
        "company"
    ]

    result = []

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        if any(
            keyword in clean.lower()
            for keyword in keywords
        ):

            result.append(clean)

    return result[:10]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🤖 AI Resume"
    )

    st.markdown(
        "## Screening System"
    )

    st.caption(
        "Smarter Hiring • Better Tomorrow"
    )

    st.divider()

    st.markdown(
        "### 🧭 Navigation"
    )

    st.write("🏠 Home")
    st.write("📄 Job Description")
    st.write("📤 Upload Resumes")
    st.write("📊 Results")
    st.write("ℹ️ About")

    st.divider()

    st.markdown(
        "### 🤖 AI Powered"
    )

    st.write(
        "Using NLP & Machine Learning "
        "for intelligent resume analysis."
    )

    st.divider()

    st.markdown(
        "### ✨ Key Features"
    )

    st.write("✓ Resume Parsing")
    st.write("✓ Skill Extraction")
    st.write("✓ TF-IDF + Cosine Similarity")
    st.write("✓ Match Score")
    st.write("✓ Candidate Ranking")

    st.divider()

    st.caption(
        '"Great teams are built with great people."'
    )


# ============================================================
# HERO
# ============================================================

with st.container(border=True):

    st.title(
        "AI Resume Screening System"
    )

    st.write(
        "Intelligent resume analysis powered by "
        "Natural Language Processing & Machine Learning"
    )

    st.write("")


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.write("")

with st.container(border=True):

    st.subheader(
        "📄  1. Job Description"
    )

    st.caption(
        "Enter the job requirements you want to "
        "match against candidate resumes."
    )

    job_description = st.text_area(
        "Job Description",
        height=170,
        placeholder=(
            "Example:\n"
            "We are looking for a Python Developer "
            "with experience in SQL, Machine Learning, "
            "Pandas, NumPy and Git."
        ),
        label_visibility="collapsed"
    )


# ============================================================
# DETECTED SKILLS
# ============================================================

if job_description.strip():

    required_skills = extract_jd_skills(
        job_description
    )

    with st.container(border=True):

        st.subheader(
            "💡 Detected Skills"
        )

        if required_skills:

            st.write(
                " • ".join(
                    required_skills
                )
            )

        else:

            st.info(
                "No predefined skills detected yet."
            )


# ============================================================
# UPLOAD
# ============================================================

st.write("")

with st.container(border=True):

    st.subheader(
        "☁️  2. Upload Candidate Resumes"
    )

    st.caption(
        "Upload one or more PDF resumes "
        "for AI-assisted screening."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:

        st.success(
            f"✓ {len(uploaded_files)} resume(s) ready for screening."
        )


# ============================================================
# SCREEN BUTTON
# ============================================================

st.write("")

screen = st.button(
    "🚀  Screen & Rank Candidates   →",
    type="primary",
    use_container_width=True
)


# ============================================================
# SCREENING
# ============================================================

if screen:

    if not job_description.strip():

        st.warning(
            "Please enter a Job Description."
        )

    elif not uploaded_files:

        st.warning(
            "Please upload at least one resume."
        )

    else:

        results = []

        with st.spinner(
            "🤖 AI is analyzing resumes..."
        ):

            for resume in uploaded_files:

                try:

                    resume_text = extract_text(
                        resume
                    )

                    score = calculate_match_score(
                        job_description,
                        resume_text
                    )

                    skills = extract_skills(
                        resume_text
                    )

                    education = extract_education(
                        resume_text
                    )

                    experience = extract_experience(
                        resume_text
                    )

                    results.append({

                        "Candidate":
                            resume.name,

                        "Score":
                            score,

                        "Skills":
                            skills,

                        "Education":
                            education,

                        "Experience":
                            experience

                    })

                except Exception as error:

                    st.error(
                        f"Error processing "
                        f"{resume.name}: {error}"
                    )


        # ====================================================
        # SORT
        # ====================================================

        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        if results:

            st.write("")

            st.subheader(
                "📊 Screening Dashboard"
            )

            total = len(results)

            average = round(
                sum(
                    r["Score"]
                    for r in results
                ) / total,
                2
            )

            top = results[0]["Score"]

            required = len(
                extract_jd_skills(
                    job_description
                )
            )


            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "👥 Candidates",
                    total
                )

            with c2:

                st.metric(
                    "⭐ Top Match",
                    f"{top}%"
                )

            with c3:

                st.metric(
                    "📈 Average",
                    f"{average}%"
                )

            with c4:

                st.metric(
                    "🛠️ Skills",
                    required
                )


            # =================================================
            # RANKINGS
            # =================================================

            st.write("")

            st.subheader(
                "🏆 Candidate Rankings"
            )


            for rank, result in enumerate(
                results,
                start=1
            ):

                score = result["Score"]


                if score >= 70:

                    status = "🟢 High Match"

                elif score >= 50:

                    status = "🟡 Moderate Match"

                else:

                    status = "🔴 Low Match"


                with st.container(
                    border=True
                ):

                    left, right = st.columns(
                        [3, 1]
                    )

                    with left:

                        st.markdown(
                            f"### 🏆 Rank {rank} — "
                            f"{result['Candidate']}"
                        )

                        st.caption(
                            "AI-assisted candidate analysis"
                        )

                    with right:

                        st.metric(
                            "Match Score",
                            f"{score}%"
                        )


                    st.progress(
                        min(
                            int(score),
                            100
                        )
                    )

                    st.write(
                        f"**Status:** {status}"
                    )

                    st.divider()


                    a, b, c = st.columns(3)


                    with a:

                        st.markdown(
                            "#### 🛠️ Skills"
                        )

                        if result["Skills"]:

                            st.write(
                                " • ".join(
                                    result["Skills"]
                                )
                            )

                        else:

                            st.caption(
                                "No skills detected."
                            )


                    with b:

                        st.markdown(
                            "#### 🎓 Education"
                        )

                        if result["Education"]:

                            for item in result["Education"]:

                                st.write(
                                    f"• {item}"
                                )

                        else:

                            st.caption(
                                "Education not detected."
                            )


                    with c:

                        st.markdown(
                            "#### 💼 Experience"
                        )

                        if result["Experience"]:

                            for item in result["Experience"]:

                                st.write(
                                    f"• {item}"
                                )

                        else:

                            st.caption(
                                "Experience not detected."
                            )


            # =================================================
            # FINISH
            # =================================================

            st.write("")

            st.success(
                "✨ Resume screening completed successfully!"
            )

            st.info(
                "This system provides AI-assisted matching "
                "to support recruiter review. Final candidate "
                "evaluation should include human review."
            )