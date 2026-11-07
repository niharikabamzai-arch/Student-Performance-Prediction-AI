import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 1. PAGE SETTINGS
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Success AI", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------
# 2. CONFIGURATION (edit these to match trainmodel.py)
# ----------------------------------------------------------------------
# Left side: name used in this app. Right side: column name used when training.
FEATURE_COLUMNS = {
    "attendance": "Attendance_Percentage",
    "internal":   "Internal_Marks",
    "assignment": "Assignment_Score",
    "previous":   "Previous_Semester_Score",
    "hours":      "Study_Hours_Per_Day",
    "engagement": "Engagement_Level",
}

# Colours and messages for each performance category
CATEGORY_STYLE = {
    "Excellent": {"color": "#22d3a6", "icon": "🌟", "msg": "Outstanding! You are performing at the top level."},
    "Good":      {"color": "#38bdf8", "icon": "👍", "msg": "Good work! A little more effort will take you higher."},
    "Average":   {"color": "#f5b94a", "icon": "📚", "msg": "Average so far. There is real room to improve."},
    "At Risk":   {"color": "#ff5d7a", "icon": "💪", "msg": "You need extra attention. Start with the plan below."},
}

# General advice cards shown for each category
CATEGORY_TIPS = {
    "Excellent": [
        ("🚀", "Aim Higher", "Take on advanced topics, projects or competitions to stretch your skills."),
        ("🤝", "Share Knowledge", "Teach peers. Explaining concepts is one of the best ways to master them."),
        ("⚖️", "Stay Balanced", "Keep your routine steady and protect your rest and wellbeing."),
    ],
    "Good": [
        ("🎯", "Target Weak Spots", "Find your two weakest topics and give them focused practice each week."),
        ("📝", "Practise Papers", "Solve previous question papers under timed conditions."),
        ("📅", "Weekly Review", "Spend 30 minutes each week reviewing what you learned."),
    ],
    "Average": [
        ("🗓️", "Build a Timetable", "Create a fixed daily study schedule and stick to it."),
        ("🙋", "Ask for Help", "Meet teachers or mentors about topics you find difficult."),
        ("🔁", "Revise Regularly", "Short daily revision beats last-minute cramming."),
    ],
    "At Risk": [
        ("🆘", "Talk to a Mentor", "Speak to a teacher or counsellor this week and share your challenges."),
        ("🧱", "Rebuild Basics", "Go back to the fundamentals and work upwards step by step."),
        ("⏰", "Small Daily Wins", "Set small, achievable daily goals and track your progress."),
    ],
}

# ----------------------------------------------------------------------
# 3. LOAD MODEL (and encoder, if saved)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    data = joblib.load("student_model.pkl")

    if isinstance(data, dict):
        model = data["model"]
        encoder = data["encoder"]
    else:
        model = data
        encoder = joblib.load("encoder.pkl")

    return model, encoder

try:
    model, encoder = load_artifacts()
except FileNotFoundError:
    st.error("Model file not found. Please run trainmodel.py first.")
    st.stop()

# ----------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ----------------------------------------------------------------------
def html(block: str):
    """Render HTML. Strips indentation so Markdown does not treat it as code."""
    st.markdown("".join(line.strip() for line in block.splitlines()), unsafe_allow_html=True)


def encode_engagement(level: str):
    """Encode Engagement Level with the saved encoder (LabelEncoder or OrdinalEncoder)."""
    if encoder is None:
        return level  # model was trained on text (unlikely) or without this feature
    try:
        return int(encoder.transform([level])[0])          # LabelEncoder style
    except Exception:
        return int(encoder.transform([[level]])[0][0])     # OrdinalEncoder style


def build_input_row(values: dict):
    """Create a one-row DataFrame whose columns match the training order exactly."""
    row = {FEATURE_COLUMNS[k]: v for k, v in values.items()}
    row[FEATURE_COLUMNS["engagement"]] = encode_engagement(values["engagement"])

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = list(row.keys())

    missing = [c for c in expected if c not in row]
    if missing:
        st.error(
            f"The model expects columns {missing} which this app does not provide. "
            "Update FEATURE_COLUMNS at the top of app.py to match trainmodel.py."
        )
        st.stop()
    unused = [c for c in row if c not in expected]
    return pd.DataFrame([row])[expected], unused


def make_suggestions(v: dict):
    """Rule-based personalised suggestions (same idea as the original app)."""
    s = []
    if v["attendance"] < 75:
        s.append(("📅", "Attendance", "Improve your attendance and attend classes regularly."))
    if v["assignment"] < 6:
        s.append(("📝", "Assignments", "Practise assignments and complete your work on time."))
    if v["internal"] < 18:
        s.append(("🧠", "Internal Marks", "Revise important topics and practise questions."))
    if v["previous"] < 50:
        s.append(("📈", "Previous Semester", "Review last semester's weak subjects and fill the gaps."))
    if v["hours"] < 2:
        s.append(("⏳", "Study Time", "Try to study at least 2 hours daily."))
    if v["engagement"] == "Low":
        s.append(("🔥", "Engagement", "Participate more in class, ask questions and join study groups."))
    return s


def metric_card(icon, label, value_text, percent):
    percent = max(0, min(100, percent))
    html(f"""
    <div class="metric-card">
      <div class="metric-top"><span class="metric-icon">{icon}</span><span class="metric-label">{label}</span></div>
      <div class="metric-value">{value_text}</div>
      <div class="bar"><div class="bar-fill" style="width:{percent}%"></div></div>
    </div>
    """)

# ----------------------------------------------------------------------
# 5. CUSTOM CSS (glassmorphism, 3D, animations, responsive)
# ----------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
:root{
  --navy:#070b1f; --navy2:#0d1433; --purple:#8b5cf6; --blue:#38bdf8; --electric:#3b82f6;
  --glass:rgba(255,255,255,0.06); --border:rgba(139,92,246,0.35); --text:#e8ecff; --muted:#9aa4d4;
}
html{scroll-behavior:smooth;}
html, body, [class*="css"]{font-family:'Poppins',sans-serif;color:var(--text);}
.stApp{
  background:radial-gradient(circle at 15% 10%, #1b1450 0%, transparent 45%),
             radial-gradient(circle at 85% 20%, #0b2a63 0%, transparent 40%),
             linear-gradient(135deg,#070b1f,#0d1433 60%,#120a33);
  background-size:140% 140%; animation:bgmove 18s ease infinite alternate;
}
@keyframes bgmove{0%{background-position:0% 0%}100%{background-position:100% 100%}}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1200px; padding-top:1.5rem; padding-bottom:4rem;}

/* ---------- Hero ---------- */
.hero{display:grid; grid-template-columns:1.1fr 0.9fr; gap:2rem; align-items:center; padding:2rem 0 3rem;}
.hero h1{font-size:clamp(2.4rem,6vw,4.4rem); font-weight:800; line-height:1.05; margin:0 0 1rem;
  background:linear-gradient(90deg,#fff,#a78bfa 50%,#38bdf8); -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p.sub{font-size:clamp(1rem,2vw,1.25rem); color:var(--muted); margin-bottom:1.4rem;}
.badge{display:inline-block; padding:.35rem .9rem; border-radius:999px; font-size:.8rem; letter-spacing:.08em;
  background:var(--glass); border:1px solid var(--border); color:#c4b5fd; margin-bottom:1rem;}
.hero-stats{display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:1rem;}
.hero-stats div{padding:.6rem 1rem; border-radius:14px; background:var(--glass); border:1px solid var(--border);
  backdrop-filter:blur(10px); font-size:.85rem; color:var(--muted);}
.hero-stats b{display:block; color:#fff; font-size:1.1rem;}

/* 3D scene */
.scene{position:relative; height:380px; perspective:900px; display:flex; align-items:center; justify-content:center;}
.orb-wrap{position:relative; width:220px; height:220px; transform-style:preserve-3d; animation:spin 16s linear infinite, floaty 6s ease-in-out infinite;}
.orb{position:absolute; inset:30px; border-radius:50%;
  background:radial-gradient(circle at 30% 28%, #c4b5fd, #6d28d9 45%, #1e1b4b 80%);
  box-shadow:0 0 60px rgba(139,92,246,.8), 0 0 140px rgba(59,130,246,.5), inset -20px -20px 50px rgba(0,0,0,.5);}
.ring{position:absolute; inset:0; border-radius:50%; border:2px solid rgba(56,189,248,.7); box-shadow:0 0 18px rgba(56,189,248,.6);}
.ring.r1{transform:rotateX(75deg);} .ring.r2{transform:rotateY(75deg); border-color:rgba(167,139,250,.7);}
.ring.r3{transform:rotateX(60deg) rotateY(40deg); inset:-18px; border-color:rgba(59,130,246,.5);}
@keyframes spin{from{transform:rotateY(0) rotateX(12deg)}to{transform:rotateY(360deg) rotateX(12deg)}}
@keyframes floaty{0%,100%{margin-top:0}50%{margin-top:-18px}}
/* Floating labels around the planet */
.holo {
    position: absolute;
    padding: .7rem 1rem;
    border-radius: 16px;

    background: rgba(10, 20, 55, 0.95);
    color: #ffffff !important;

    border: 1px solid rgba(56,189,248,.7);
    backdrop-filter: blur(12px);

    font-size: .85rem;
    font-weight: 600;
    text-shadow: 0 0 8px rgba(255,255,255,.4);

    box-shadow: 0 10px 30px rgba(0,0,0,.5);

    z-index: 10;
    animation: floaty 5s ease-in-out infinite;
    transition: transform .3s;
}

.holo:hover {
    transform: translateZ(40px) scale(1.08);
}

.holo.h1 {
    top: 8%;
    left: 2%;
}

.holo.h2 {
    top: 62%;
    left: 0%;
    animation-delay: 1s;
}

.holo.h3 {
    top: 12%;
    right: 0%;
    animation-delay: 2s;
}

.holo.h4 {
    top: 68%;
    right: 4%;
    animation-delay: 3s;
}

/* Keep planet behind the labels */
.orb-wrap {
    z-index: 2;
}

.particle {
    z-index: 0;
    pointer-events: none;
}
.particle{position:absolute; width:6px; height:6px; border-radius:50%; background:#7dd3fc; box-shadow:0 0 12px #38bdf8; animation:rise 7s linear infinite;}
@keyframes rise{0%{transform:translateY(40px); opacity:0}20%{opacity:1}100%{transform:translateY(-300px); opacity:0}}

/* ---------- Section titles ---------- */
.section-title{font-size:1.6rem; font-weight:700; margin:2.5rem 0 .3rem;}
.section-sub{color:var(--muted); margin-bottom:1rem;}
.grad{background:linear-gradient(90deg,#a78bfa,#38bdf8); -webkit-background-clip:text; background-clip:text; color:transparent;}

/* ---------- Glass containers for inputs ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--glass); border:1px solid var(--border)!important; border-radius:22px!important;
  backdrop-filter:blur(14px); box-shadow:0 18px 50px rgba(0,0,0,.35), inset 0 0 30px rgba(139,92,246,.06);
  transition:transform .3s, box-shadow .3s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-4px); box-shadow:0 24px 60px rgba(59,130,246,.25);}
label, .stSlider label p, .stSelectbox label p, .stNumberInput label p{color:#dbe2ff!important; font-weight:500;}
/* Dropdown and percentage input styling */
div[data-baseweb="select"] > div {
    background: #151d42 !important;
    color: white !important;
}

div[data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #151d42 !important;
    border: 1px solid #8b5cf6 !important;
    border-radius: 12px !important;
}

/* Make semester score number clearly visible */
div[data-testid="stNumberInput"] input,
div[data-testid="stNumberInput"] input[type="number"] {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
    background: #f1f5f9 !important;
}
div[data-baseweb="slider"] [role="slider"]{background:var(--blue)!important; box-shadow:0 0 14px var(--blue);}

/* ---------- Buttons ---------- */
.stButton > button{
  width:100%; border:none; border-radius:16px; padding:.9rem 1.5rem; font-weight:600; font-size:1.05rem; color:#fff;
  background:linear-gradient(90deg,#7c3aed,#3b82f6); box-shadow:0 8px 0 #3b1d8f, 0 18px 40px rgba(124,58,237,.55);
  animation:glow 2.6s ease-in-out infinite; transition:transform .15s, box-shadow .15s;}
.stButton > button:hover{transform:translateY(-3px); color:#fff;}
.stButton > button:active{transform:translateY(5px); box-shadow:0 3px 0 #3b1d8f, 0 8px 20px rgba(124,58,237,.5);}
@keyframes glow{0%,100%{filter:brightness(1)}50%{filter:brightness(1.18)}}
.stDownloadButton > button{border-radius:14px; background:var(--glass); color:#fff; border:1px solid var(--border); backdrop-filter:blur(10px);}
.stDownloadButton > button:hover{border-color:var(--blue); color:#fff; transform:translateY(-2px);}

/* ---------- Result card ---------- */
.result-wrap{perspective:1000px;}
.result-card{display:grid; grid-template-columns:auto 1fr; gap:2rem; align-items:center; padding:2rem; border-radius:26px;
  background:var(--glass); border:1px solid var(--accent); backdrop-filter:blur(16px);
  box-shadow:0 0 50px color-mix(in srgb, var(--accent) 35%, transparent), 0 20px 60px rgba(0,0,0,.4);
  transition:transform .35s; animation:pop .7s ease;}
.result-card:hover{transform:rotateX(4deg) rotateY(-4deg) translateZ(10px);}
@keyframes pop{from{opacity:0; transform:translateY(30px) scale(.96)}to{opacity:1; transform:none}}
.ring-box{position:relative; width:150px; height:150px;}
.ring-box svg{transform:rotate(-90deg);}
.ring-bg{stroke:rgba(255,255,255,.1);} .ring-fg{stroke:var(--accent); stroke-linecap:round; stroke-dasharray:339.3;
  filter:drop-shadow(0 0 8px var(--accent)); animation:fill 1.6s ease forwards;}
@keyframes fill{from{stroke-dashoffset:339.3}to{stroke-dashoffset:var(--off)}}
.ring-text{position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
.ring-text b{font-size:1.9rem;} .ring-text span{font-size:.7rem; color:var(--muted);}
.cat{font-size:clamp(1.8rem,4vw,2.8rem); font-weight:800; color:var(--accent); text-shadow:0 0 25px var(--accent);}
.cat-msg{color:var(--muted);}

/* ---------- Cards ---------- */
.card-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.tip-card, .metric-card{padding:1.3rem; border-radius:20px; background:var(--glass); border:1px solid var(--border);
  backdrop-filter:blur(12px); transition:transform .3s, box-shadow .3s, border-color .3s;}
/* ---------- TEXT VISIBILITY FIX ---------- */

/* Main section headings */
.section-title {
    color: #ffffff !important;
}

/* Confidence percentage */
.ring-text b {
    color: #ffffff !important;
}

.ring-text span {
    color: #dbeafe !important;
}

/* Predicted performance description */
.cat-msg {
    color: #dbeafe !important;
}

/* Improvement plan card headings */
.tip-card h4 {
    color: #ffffff !important;
}

/* Improvement plan descriptions */
.tip-card p {
    color: #dbeafe !important;
}

/* Academic summary text */
.metric-label,
.metric-value {
    color: #ffffff !important;
}
.tip-icon{font-size:1.8rem;} .tip-card h4{margin:.5rem 0 .3rem; font-size:1.05rem;} .tip-card p{color:var(--muted); font-size:.9rem; margin:0;}
.metric-top{display:flex; align-items:center; gap:.5rem; color:var(--muted); font-size:.85rem;}
.metric-value{font-size:1.7rem; font-weight:700; margin:.4rem 0 .7rem;}
.bar{height:8px; border-radius:99px; background:rgba(255,255,255,.1); overflow:hidden;}
.bar-fill{height:100%; border-radius:99px; background:linear-gradient(90deg,#8b5cf6,#38bdf8); box-shadow:0 0 10px #38bdf8;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{background:rgba(10,14,40,.85); border-right:1px solid var(--border); backdrop-filter:blur(16px);}
.side-card{padding:1rem; border-radius:16px; background:var(--glass); border:1px solid var(--border); margin-bottom:.9rem; font-size:.88rem; color:var(--muted);}
.side-card h4{color:#fff; margin:0 0 .4rem; font-size:1rem;}
.chip{display:inline-block; padding:.2rem .6rem; margin:.15rem; border-radius:999px; background:rgba(139,92,246,.18); border:1px solid var(--border); color:#ddd6fe; font-size:.75rem;}

/* ---------- Responsive ---------- */
@media (max-width:992px){ .hero{grid-template-columns:1fr;} .scene{height:320px;} .card-grid{grid-template-columns:repeat(2,1fr);} }
@media (max-width:640px){ .card-grid{grid-template-columns:1fr;} .result-card{grid-template-columns:1fr; text-align:center; justify-items:center;} .holo{font-size:.7rem; padding:.5rem .7rem;} }
@media (prefers-reduced-motion:reduce){ *{animation:none!important;} }
</style>
""")

# ----------------------------------------------------------------------
# 6. SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    html("""
    <div class="side-card"><h4>🎓 About the Project</h4>
    Student Success AI predicts academic performance from a student's details and gives a personalised improvement plan.</div>
    <div class="side-card"><h4>🌳 Algorithm</h4>Machine learning model: <b style="color:#fff">Decision Tree Classifier</b></div>
    <div class="side-card"><h4>🧩 Input Features</h4>
    <span class="chip">Attendance</span><span class="chip">Internal Marks</span><span class="chip">Assignment Score</span>
    <span class="chip">Previous Semester</span><span class="chip">Study Hours</span><span class="chip">Engagement Level</span></div>
    <div class="side-card"><h4>ℹ️ Project Info</h4>Built with Python, Streamlit, scikit-learn and pandas.<br>
    For educational purposes. Trained on a small sample dataset.</div>
    """)

# ----------------------------------------------------------------------
# 7. HERO SECTION
# ----------------------------------------------------------------------
particles = "".join(
    f'<span class="particle" style="left:{x}%; bottom:{y}%; animation-delay:{d}s"></span>'
    for x, y, d in [(12, 10, 0), (30, 5, 1.5), (48, 15, 3), (66, 8, 4.5), (82, 12, 2), (92, 6, 5.5), (22, 20, 6)]
)

html(f"""
<div class="hero">
  <div>
    <span class="badge">✦ AI-POWERED • DECISION TREE MODEL</span>
    <h1>Student Success AI</h1>
    <p class="sub">Your Personal AI-Powered Academic Performance Assistant</p>
    <div class="hero-stats">
      <div><b>6</b>Input features</div><div><b>4</b>Performance levels</div><div><b>Instant</b>Predictions</div>
    </div>
  </div>
  <div class="scene">
    {particles}
    <div class="holo h1">🎓 Academic Insights</div><div class="holo h2">📊 Performance Score</div>
    <div class="holo h3">🧠 AI Analysis</div><div class="holo h4">🚀 Success Plan</div>
    <div class="orb-wrap"><div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div><div class="orb"></div></div>
  </div>
</div>
""")

# ----------------------------------------------------------------------
# 8. ACADEMIC DETAILS (inputs)
# ----------------------------------------------------------------------
html('<div class="section-title">📋 Your <span class="grad">Academic Details</span></div>'
     '<div class="section-sub">Adjust the controls below, then run the prediction.</div>')

left, right = st.columns(2)
with left:
    with st.container(border=True):
        attendance = st.slider("Attendance Percentage (%)", 0, 100, 85)
        internal = st.slider("Internal Marks (out of 30)", 0, 30, 25)
        assignment = st.slider("Assignment Score (out of 10)", 0, 10, 8)
with right:
    with st.container(border=True):
        previous = st.number_input("Previous Semester Score (%)", 0.0, 100.0, 70.0, step=1.0)
        hours = st.slider("Daily Study Hours", 0, 12, 4)
        engagement = st.selectbox("Engagement Level", ["High", "Medium", "Low"], index=1)

st.write("")
predict_clicked = st.button("🔍 Predict My Performance", type="primary")

values = {
    "attendance": attendance, "internal": internal, "assignment": assignment,
    "previous": previous, "hours": hours, "engagement": engagement,
}

# ----------------------------------------------------------------------
# 9. PREDICTION (saved in session_state so the download button does not erase it)
# ----------------------------------------------------------------------
if predict_clicked:
    student, unused = build_input_row(values)
    prediction = str(model.predict(student)[0])
    probabilities = model.predict_proba(student)[0]
    confidence = float(max(probabilities)) * 100
    class_probs = dict(zip([str(c) for c in model.classes_], probabilities))
    st.session_state["result"] = {
        "values": values.copy(), "prediction": prediction,
        "confidence": confidence, "class_probs": class_probs, "unused": unused,
    }

result = st.session_state.get("result")

if result:
    v = result["values"]
    prediction = result["prediction"]
    confidence = result["confidence"]
    style = CATEGORY_STYLE.get(prediction, CATEGORY_STYLE["Average"])
    offset = 339.3 * (1 - confidence / 100)

    # ---- AI prediction card ----
    html('<div class="section-title">🤖 AI <span class="grad">Prediction</span></div>')
    html(f"""
    <div class="result-wrap"><div class="result-card" style="--accent:{style['color']}; --off:{offset:.1f}">
      <div class="ring-box">
        <svg width="150" height="150" viewBox="0 0 120 120">
          <circle class="ring-bg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
          <circle class="ring-fg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
        </svg>
        <div class="ring-text"><b>{confidence:.1f}%</b><span>CONFIDENCE</span></div>
      </div>
      <div><div class="cat-msg">Predicted Performance</div>
      <div class="cat">{style['icon']} {prediction}</div>
      <div class="cat-msg">{style['msg']}</div></div>
    </div></div>
    """)

    if result["unused"]:
        st.info(f"Note: your saved model was trained without {result['unused']}, so those inputs are not used in the prediction. "
                "Retrain with all six features to include them.")

    # ---- Improvement plan ----
    html('<div class="section-title">💡 Academic <span class="grad">Improvement Plan</span></div>'
         '<div class="section-sub">Recommendations for your predicted category.</div>')
    tips = "".join(
        f'<div class="tip-card"><div class="tip-icon">{i}</div><h4>{t}</h4><p>{d}</p></div>'
        for i, t, d in CATEGORY_TIPS.get(prediction, [])
    )
    html(f'<div class="card-grid">{tips}</div>')

    suggestions = make_suggestions(v)
    html('<div class="section-sub" style="margin-top:1.2rem">Personalised focus areas based on your inputs:</div>')
    if suggestions:
        cards = "".join(
            f'<div class="tip-card"><div class="tip-icon">{i}</div><h4>{t}</h4><p>{d}</p></div>'
            for i, t, d in suggestions
        )
        html(f'<div class="card-grid">{cards}</div>')
    else:
        st.success("You are meeting all the basic targets. Keep practising to maintain your performance!")

    # ---- Academic summary ----
    html('<div class="section-title">📊 Academic <span class="grad">Summary</span></div>')
    cols = st.columns(3)
    summary = [
        ("📅", "Attendance", f"{v['attendance']}%", v["attendance"]),
        ("📝", "Assignment Score", f"{v['assignment']}/10", v["assignment"] * 10),
        ("🧠", "Internal Marks", f"{v['internal']}/30", v["internal"] / 30 * 100),
        ("📈", "Previous Semester", f"{v['previous']:.0f}%", v["previous"]),
        ("⏳", "Daily Study Hours", f"{v['hours']} hrs", v["hours"] / 12 * 100),
        ("🔥", "Engagement Level", v["engagement"], {"High": 100, "Medium": 60, "Low": 25}[v["engagement"]]),
    ]
    for idx, (icon, label, text, pct) in enumerate(summary):
        with cols[idx % 3]:
            metric_card(icon, label, text, pct)
            st.write("")

    # ---- Download CSV report ----
    report_suggestions = [d for _, _, d in suggestions] or ["Keep practising to maintain your performance!"]
    report = pd.DataFrame([{
        "Attendance": v["attendance"], "Assignment_Score": v["assignment"], "Internal_Marks": v["internal"],
        "Previous_Semester_Score": v["previous"], "Study_Hours": v["hours"], "Engagement_Level": v["engagement"],
        "Predicted_Performance": prediction, "Confidence_Percent": round(confidence, 1),
        "Suggestions": " ".join(report_suggestions),
    }])
    st.download_button("📥 Download My Report", data=report.to_csv(index=False),
                       file_name="student_performance_report.csv", mime="text/csv")

    st.caption("Note: This prediction is based on a small sample dataset and is intended for educational purposes.")