import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 1. PAGE SETTINGS
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Success AI", layout="wide")

# ----------------------------------------------------------------------
# 2. CONFIGURATION (edit these to match trainmodel.py)
# ----------------------------------------------------------------------
FEATURE_COLUMNS = {
    "attendance": "Attendance_Percentage",
    "internal":   "Internal_Marks",
    "assignment": "Assignment_Score",
    "previous":   "Previous_Semester_Score",
    "hours":      "Study_Hours_Per_Day",
    "engagement": "Engagement_Level",
}

CATEGORY_STYLE = {
    "Excellent": {"color": "#2e7d4f", "msg": "Outstanding. You are performing at the top level."},
    "Good":      {"color": "#6f8a2a", "msg": "Good work. A little more effort will take you higher."},
    "Average":   {"color": "#b97a12", "msg": "Average so far. There is real room to improve."},
    "At Risk":   {"color": "#a8382a", "msg": "You need extra attention. Start with the plan below."},
}

CATEGORY_TIPS = {
    "Excellent": [
        ("Aim higher", "Take on advanced topics, projects or competitions to stretch your skills."),
        ("Share knowledge", "Teach peers. Explaining concepts is one of the best ways to master them."),
        ("Stay balanced", "Keep your routine steady and protect your rest and wellbeing."),
    ],
    "Good": [
        ("Target weak spots", "Find your two weakest topics and give them focused practice each week."),
        ("Practise papers", "Solve previous question papers under timed conditions."),
        ("Weekly review", "Spend 30 minutes each week reviewing what you learned."),
    ],
    "Average": [
        ("Build a timetable", "Create a fixed daily study schedule and stick to it."),
        ("Ask for help", "Meet teachers or mentors about topics you find difficult."),
        ("Revise regularly", "Short daily revision beats last-minute cramming."),
    ],
    "At Risk": [
        ("Talk to a mentor", "Speak to a teacher or counsellor this week and share your challenges."),
        ("Rebuild the basics", "Go back to the fundamentals and work upwards step by step."),
        ("Set small daily goals", "Choose achievable daily targets and track your progress."),
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
    if encoder is None:
        return level
    try:
        return int(encoder.transform([level])[0])
    except Exception:
        return int(encoder.transform([[level]])[0][0])


def build_input_row(values: dict):
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
    s = []
    if v["attendance"] < 75:
        s.append(("Attendance", "Improve your attendance and attend classes regularly."))
    if v["assignment"] < 6:
        s.append(("Assignments", "Practise assignments and complete your work on time."))
    if v["internal"] < 18:
        s.append(("Internal marks", "Revise important topics and practise questions."))
    if v["previous"] < 50:
        s.append(("Previous semester", "Review last semester's weak subjects and fill the gaps."))
    if v["hours"] < 2:
        s.append(("Study time", "Try to study at least 2 hours daily."))
    if v["engagement"] == "Low":
        s.append(("Engagement", "Participate more in class, ask questions and join study groups."))
    return s


def metric_card(label, value_text, percent):
    percent = max(0, min(100, percent))
    html(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value_text}</div>
      <div class="bar"><div class="bar-fill" style="width:{percent}%"></div></div>
    </div>
    """)


def tip_cards(items):
    return "".join(f'<div class="tip-card"><h4>{t}</h4><p>{d}</p></div>' for t, d in items)


def pillars(items, w=40, gap=26, big=False):
    """3D column chart. items = [(label or None, percent 0-100, value text or None)]."""
    out = []
    top = max(p for _, p, _ in items)
    for i, (label, pct, val) in enumerate(items):
        peak = " peak" if pct == top else ""
        h = int(30 + max(0, min(100, pct)) * 1.5)
        v = f'<div class="val">{val}</div>' if val else ""
        lab = f'<div class="lab">{label}</div>' if label else ""
        out.append(
            f'<div class="pillar{peak}" style="--h:{h}px; --d:{0.15 * i:.2f}s">'
            f'<div class="body"><div class="pface pf"></div><div class="pface pr"></div><div class="pface pt"></div>{v}</div>'
            f'{lab}</div>'
        )
    cls = "iso-wrap big" if big else "iso-wrap"
    return (f'<div class="{cls}"><div class="iso" style="--w:{w}px; --gap:{gap}px">' + "".join(out)
            + '<div class="pl-top"></div><div class="pl-front"></div><div class="pl-right"></div></div></div>')

# ----------------------------------------------------------------------
# 5. CUSTOM CSS
# ----------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@500;600;700&display=swap');
:root{
  --bg:#e4e4e5; --panel:#fafafa; --panel2:#f0f0f1; --line:#c9c9cc; --ink:#1d1d20; --muted:#5b5b60;
  --green:#7a1f2e; --green-dark:#4a1119; --green-soft:#ecdcdf;
  --lift:0 1px 0 rgba(255,255,255,.9) inset, 0 1px 2px rgba(20,20,22,.10), 0 8px 16px -6px rgba(20,20,22,.18), 0 26px 40px -22px rgba(20,20,22,.30);
}
html{scroll-behavior:smooth;}
html, body, [class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background:radial-gradient(rgba(60,60,66,.10) 1px,transparent 1.3px) 0 0/24px 24px, linear-gradient(180deg,#ececed 0%,#e4e4e5 40%,#d4d4d6 100%); color:var(--ink);}
.stApp p, .stApp span, .stApp li{color:inherit;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1180px; padding-top:2rem; padding-bottom:4rem;}
h1,h2,h3,.serif{font-family:'Source Serif 4',serif;}

.hero{display:grid; grid-template-columns:1.1fr .9fr; gap:2rem; align-items:center; padding:2.6rem 3rem; margin-bottom:.5rem; border-radius:20px; border:1px solid #0a0a0b; background:radial-gradient(circle at 80% 30%,#6a2a36 0%,transparent 55%),linear-gradient(135deg,#2e2e30,#161617); box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 20px 30px -12px rgba(0,0,0,.55),0 50px 70px -35px rgba(0,0,0,.6);}
.hero h1{font-size:clamp(2.4rem,5.4vw,4rem); font-weight:700; line-height:1.1; margin:0 0 1.2rem; color:#fff!important; letter-spacing:.005em; text-shadow:1px 1px 0 #b8505f,2px 2px 0 #a2434f,3px 3px 0 #8a3641,4px 4px 0 #6f2a33,5px 5px 0 #5a212a,6px 6px 0 #45171e,8px 14px 18px rgba(0,0,0,.5);}
.hero p.sub{font-size:1.1rem; color:#d6d6d9; max-width:34rem; margin:0 0 1.8rem; line-height:1.65;}
.facts{display:flex; gap:0; border:1px solid rgba(255,255,255,.2); border-radius:12px; background:rgba(255,255,255,.07); box-shadow:0 10px 24px -10px rgba(0,0,0,.5); width:fit-content; overflow:hidden;}
.facts div{padding:.8rem 1.4rem; font-size:.85rem; color:#b8b8bd; border-right:1px solid rgba(255,255,255,.18);}
.facts div:last-child{border-right:none;}
.facts b{display:block; font-family:'Source Serif 4',serif; color:#e6c36a; font-size:1.5rem; font-weight:600;}

.iso-wrap{perspective:1500px; display:flex; justify-content:center; padding:70px 0 80px;}
.iso{position:relative; display:flex; align-items:flex-end; gap:var(--gap); height:180px; transform-style:preserve-3d; transform:rotateX(-22deg) rotateY(-32deg);}
.pillar{position:relative; width:var(--w); height:var(--h); transform-style:preserve-3d;}
.body{position:absolute; inset:0; transform-style:preserve-3d; transform-origin:50% 100%; animation:rise 1.1s cubic-bezier(.2,.8,.2,1) var(--d) both;}
@keyframes rise{from{transform:scaleY(.03)}to{transform:scaleY(1)}}
.pface{position:absolute; left:0;}
.pf{top:0; width:var(--w); height:100%; transform:translateZ(calc(var(--w) / 2)); background:linear-gradient(180deg,#d65a6e,#9a2a3e);}
.pr{top:0; width:var(--w); height:100%; transform:rotateY(90deg) translateZ(calc(var(--w) / 2)); background:linear-gradient(180deg,#9a2a3e,#5e1823);}
.pt{top:calc(var(--w) / -2); width:var(--w); height:var(--w); transform:rotateX(90deg); background:#f0a0ac;}
.val{position:absolute; top:-30px; left:-20px; width:calc(var(--w) + 40px); text-align:center; font-size:13px; font-weight:600; color:var(--ink); transform:translateZ(calc(var(--w) / 2));}
.pillar.peak .pf{background:linear-gradient(180deg,#ebc96f,#b08a2e);}
.pillar.peak .pr{background:linear-gradient(180deg,#b08a2e,#7a5c18);}
.pillar.peak .pt{background:#f4e0a2;}
.iso-wrap .val{color:#f2f2f3; text-shadow:0 1px 2px rgba(0,0,0,.5);}
.iso-wrap.big{margin-top:.6rem; border-radius:20px; border:1px solid #0a0a0b; background:radial-gradient(circle at 50% 20%,#6a2a36 0%,transparent 60%),linear-gradient(135deg,#2e2e30,#161617); box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 20px 30px -12px rgba(0,0,0,.55),0 50px 70px -35px rgba(0,0,0,.6);}
.lab{position:absolute; left:-30px; top:calc(100% + 3px); width:calc(var(--w) + 60px); height:18px; line-height:18px; text-align:center; font-size:12px; font-weight:500; color:#f2f2f3; transform:translateZ(61px);}
.pl-top{position:absolute; left:-36px; right:-36px; bottom:-60px; height:120px; transform:rotateX(90deg); background:linear-gradient(180deg,#dcdcde,#c3c3c6); border:1px solid rgba(0,0,0,.12);}
.pl-front{position:absolute; left:-36px; right:-36px; top:100%; height:18px; transform:translateZ(60px); background:linear-gradient(180deg,#5a5a5e,#3e3e42);}
.pl-right{position:absolute; right:-36px; top:100%; width:120px; height:18px; transform-origin:100% 0; transform:translateZ(-60px) rotateY(90deg); background:#2e2e31;}
@media (max-width:640px){ .iso-wrap.big{transform:scale(.5); transform-origin:top center; margin-bottom:-170px;} }

.section-title{display:flex; align-items:center; gap:.8rem; font-family:'Source Serif 4',serif; font-size:1.7rem; font-weight:600; margin:2.8rem 0 .4rem; color:var(--ink);}
.section-title::before{content:""; width:12px; height:1.6rem; border-radius:3px; background:linear-gradient(90deg,#c0606d,#4a1119); box-shadow:3px 4px 0 rgba(20,20,22,.22);}
.section-sub{color:var(--muted); margin-bottom:1rem; font-size:.98rem;}

div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel); border:1px solid var(--line)!important; border-radius:12px!important; box-shadow:var(--lift);}
label, .stSlider label p, .stSelectbox label p, .stNumberInput label p{color:var(--ink)!important; font-weight:500;}
div[data-baseweb="select"] > div{background:#fff!important; color:var(--ink)!important; border:1px solid var(--line)!important; border-radius:8px!important; box-shadow:inset 0 2px 3px rgba(20,20,22,.08);}
div[data-baseweb="select"] *{color:var(--ink)!important;}
div[data-testid="stNumberInput"] [data-baseweb="input"]{background:#fff!important; border:1px solid var(--line)!important; border-radius:8px!important; box-shadow:inset 0 2px 3px rgba(20,20,22,.08);}
div[data-testid="stNumberInput"] input{color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important; background:#fff!important; font-weight:600!important;}
div[data-testid="stNumberInput"] button{background:var(--panel2)!important; color:var(--ink)!important;}
div[data-baseweb="slider"] [role="slider"]{background:var(--green)!important; border:2px solid #fff!important; box-shadow:0 2px 5px rgba(0,0,0,.35)!important;}
div[data-testid="stSliderThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"]{color:var(--ink)!important;}

.stButton > button{width:100%; border:1px solid var(--green-dark); border-radius:10px; padding:.85rem 1.5rem; font-weight:600; font-size:1.02rem; color:#fff!important;
  background:linear-gradient(180deg,#98303f,#7a1f2e); box-shadow:0 1px 0 rgba(255,255,255,.25) inset, 0 5px 0 var(--green-dark), 0 14px 22px -6px rgba(20,20,22,.45); transition:transform .12s, box-shadow .12s;}
.stButton > button:hover{transform:translateY(-2px); color:#fff!important; border-color:var(--green-dark);}
.stButton > button:active{transform:translateY(4px); box-shadow:0 1px 0 var(--green-dark), 0 4px 8px rgba(20,20,22,.35);}
.stButton > button p{color:#fff!important;}
.stDownloadButton > button{border-radius:10px; background:var(--panel); color:var(--ink)!important; border:1px solid var(--line); box-shadow:var(--lift);}
.stDownloadButton > button:hover{border-color:var(--green); color:var(--green)!important;}
.stDownloadButton > button p{color:inherit!important;}

.result-wrap{perspective:1100px;}
.result-card{display:grid; grid-template-columns:auto 1fr; gap:2rem; align-items:center; padding:2.2rem 2.4rem; border-radius:16px; background:linear-gradient(135deg,#ffffff,color-mix(in srgb,var(--accent) 9%,#f6f6f6)); border:1px solid var(--line); border-left:10px solid var(--accent); box-shadow:var(--lift),0 30px 50px -30px var(--accent); transition:transform .35s;}
.result-card:hover{transform:rotateX(3deg) rotateY(-3deg) translateZ(8px);}
.ring-box{position:relative; width:150px; height:150px;}
.ring-box svg{transform:rotate(-90deg);}
.ring-bg{stroke:#dddddf;}
.ring-fg{stroke:var(--accent); stroke-linecap:round; stroke-dasharray:339.3; animation:fill 1.4s ease forwards;}
@keyframes fill{from{stroke-dashoffset:339.3}to{stroke-dashoffset:var(--off)}}
.ring-text{position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
.ring-text b{font-family:'Source Serif 4',serif; font-size:1.9rem; color:var(--ink)!important;}
.ring-text span{font-size:.72rem; color:var(--muted)!important;}
.cat{font-family:'Source Serif 4',serif; font-size:clamp(2rem,4vw,2.9rem); font-weight:700; color:var(--accent); line-height:1.1; margin:.2rem 0 .4rem;}
.cat-msg{color:var(--muted)!important;}

.card-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.tip-card, .metric-card{padding:1.2rem 1.3rem; border-radius:12px; background:var(--panel); border:1px solid var(--line); box-shadow:var(--lift); transition:transform .25s, box-shadow .25s;}
.tip-card:hover, .metric-card:hover{transform:perspective(700px) rotateX(5deg) translateY(-6px); box-shadow:0 1px 0 rgba(255,255,255,.9) inset, 0 14px 22px -8px rgba(20,20,22,.3), 0 36px 48px -24px rgba(20,20,22,.38);}
.tip-card{border-top:4px solid var(--green);}
.tip-card h4{margin:0 0 .4rem; font-size:1.05rem; font-weight:600; color:var(--ink)!important;}
.tip-card p{color:var(--muted)!important; font-size:.92rem; line-height:1.5; margin:0;}
.metric-label{color:var(--muted)!important; font-size:.88rem;}
.metric-value{font-family:'Source Serif 4',serif; font-size:1.8rem; font-weight:600; margin:.3rem 0 .8rem; color:var(--ink)!important;}
.bar{height:9px; border-radius:99px; background:#d6d6d8; box-shadow:inset 0 2px 3px rgba(20,20,22,.18); overflow:hidden;}
.bar-fill{height:100%; border-radius:99px; background:linear-gradient(180deg,#b8505f,#7a1f2e);}

section[data-testid="stSidebar"]{background:#d8d8da; border-right:1px solid var(--line);}
.side-card{padding:1rem 1.1rem; border-radius:10px; background:var(--panel); border:1px solid var(--line); box-shadow:var(--lift); margin-bottom:1rem; font-size:.88rem; color:var(--muted);}
.side-card h4{color:var(--ink); margin:0 0 .4rem; font-size:1rem; font-weight:600;}
.chip{display:inline-block; padding:.2rem .6rem; margin:.15rem; border-radius:6px; background:var(--green-soft); border:1px solid var(--line); color:var(--green-dark); font-size:.76rem;}

@media (max-width:992px){ .hero{grid-template-columns:1fr;} .card-grid{grid-template-columns:repeat(2,1fr);} }
@media (max-width:640px){ .card-grid{grid-template-columns:1fr;} .result-card{grid-template-columns:1fr; justify-items:center; text-align:center;} .facts{flex-direction:column; width:100%;} .facts div{border-right:none; border-bottom:1px solid var(--line);} }
@media (prefers-reduced-motion:reduce){ *{animation:none!important; transition:none!important;} }
</style>
""")

# ----------------------------------------------------------------------
# 6. SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    html("""
    <div class="side-card"><h4>About the project</h4>
    Student Success AI predicts academic performance from a student's details and gives a personalised improvement plan.</div>
    <div class="side-card"><h4>Algorithm</h4>Decision Tree Classifier</div>
    <div class="side-card"><h4>Input features</h4>
    <span class="chip">Attendance</span><span class="chip">Internal Marks</span><span class="chip">Assignment Score</span>
    <span class="chip">Previous Semester</span><span class="chip">Study Hours</span><span class="chip">Engagement Level</span></div>
    <div class="side-card"><h4>Project info</h4>Built with Python, Streamlit, scikit-learn and pandas.<br>
    For educational purposes. Trained on a small sample dataset.</div>
    """)

# ----------------------------------------------------------------------
# 7. HERO SECTION
# ----------------------------------------------------------------------
html(f"""
<div class="hero">
  <div>
    <h1>Student Success AI</h1>
    <p class="sub">Enter your attendance, marks and study habits to get a predicted performance level and a practical plan for improving it.</p>
    <div class="facts">
      <div><b>6</b>input features</div><div><b>4</b>performance levels</div><div><b>Instant</b>results</div>
    </div>
  </div>
  <div>{pillars([(None, p, None) for p in (55, 78, 64, 92, 72, 100)])}</div>
</div>
""")

# ----------------------------------------------------------------------
# 8. ACADEMIC DETAILS (inputs)
# ----------------------------------------------------------------------
html('<div class="section-title">Academic details</div>'
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
predict_clicked = st.button("Predict my performance", type="primary")

values = {
    "attendance": attendance, "internal": internal, "assignment": assignment,
    "previous": previous, "hours": hours, "engagement": engagement,
}

# ----------------------------------------------------------------------
# 9. PREDICTION
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

    html('<div class="section-title">Prediction</div>')
    html(f"""
    <div class="result-wrap"><div class="result-card" style="--accent:{style['color']}; --off:{offset:.1f}">
      <div class="ring-box">
        <svg width="150" height="150" viewBox="0 0 120 120">
          <circle class="ring-bg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
          <circle class="ring-fg" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
        </svg>
        <div class="ring-text"><b>{confidence:.1f}%</b><span>Confidence</span></div>
      </div>
      <div><div class="cat-msg">Predicted performance</div>
      <div class="cat">{prediction}</div>
      <div class="cat-msg">{style['msg']}</div></div>
    </div></div>
    """)

    if result["unused"]:
        st.info(f"Note: your saved model was trained without {result['unused']}, so those inputs are not used in the prediction. "
                "Retrain with all six features to include them.")

    html('<div class="section-title">Academic improvement plan</div>'
         '<div class="section-sub">Recommendations for your predicted category.</div>')
    html(f'<div class="card-grid">{tip_cards(CATEGORY_TIPS.get(prediction, []))}</div>')

    suggestions = make_suggestions(v)
    html('<div class="section-sub" style="margin-top:1.4rem">Personalised focus areas based on your inputs:</div>')
    if suggestions:
        html(f'<div class="card-grid">{tip_cards(suggestions)}</div>')
    else:
        st.success("You are meeting all the basic targets. Keep practising to maintain your performance.")

    html('<div class="section-title">Score profile</div>'
         '<div class="section-sub">Each column shows how close you are to the full score in that area.</div>')
    profile = [
        ("Attendance", v["attendance"], f"{v['attendance']}%"),
        ("Internal", v["internal"] / 30 * 100, f"{v['internal']}/30"),
        ("Assignment", v["assignment"] * 10, f"{v['assignment']}/10"),
        ("Previous", v["previous"], f"{v['previous']:.0f}%"),
        ("Study hrs", v["hours"] / 12 * 100, f"{v['hours']} hrs"),
        ("Engagement", {"High": 100, "Medium": 60, "Low": 25}[v["engagement"]], v["engagement"]),
    ]
    html(pillars(profile, w=54, gap=64, big=True))

    html('<div class="section-title">Academic summary</div>')
    cols = st.columns(3)
    summary = [
        ("Attendance", f"{v['attendance']}%", v["attendance"]),
        ("Assignment Score", f"{v['assignment']}/10", v["assignment"] * 10),
        ("Internal Marks", f"{v['internal']}/30", v["internal"] / 30 * 100),
        ("Previous Semester", f"{v['previous']:.0f}%", v["previous"]),
        ("Daily Study Hours", f"{v['hours']} hrs", v["hours"] / 12 * 100),
        ("Engagement Level", v["engagement"], {"High": 100, "Medium": 60, "Low": 25}[v["engagement"]]),
    ]
    for idx, (label, text, pct) in enumerate(summary):
        with cols[idx % 3]:
            metric_card(label, text, pct)
            st.write("")

    report_suggestions = [d for _, d in suggestions] or ["Keep practising to maintain your performance."]
    report = pd.DataFrame([{
        "Attendance": v["attendance"], "Assignment_Score": v["assignment"], "Internal_Marks": v["internal"],
        "Previous_Semester_Score": v["previous"], "Study_Hours": v["hours"], "Engagement_Level": v["engagement"],
        "Predicted_Performance": prediction, "Confidence_Percent": round(confidence, 1),
        "Suggestions": " ".join(report_suggestions),
    }])
    st.download_button("Download my report", data=report.to_csv(index=False),
                       file_name="student_performance_report.csv", mime="text/csv")

    st.caption("Note: This prediction is based on a small sample dataset and is intended for educational purposes.")
