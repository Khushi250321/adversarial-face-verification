import streamlit as st
import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, r'C:\transferattacktInterns\core')

st.set_page_config(
    page_title="FaceGuard — Adversarial Attack Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 28px;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: white;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Success / fail badge */
    .badge-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border-radius: 30px;
        padding: 8px 22px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-fail {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a1a;
        border-radius: 30px;
        padding: 8px 22px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* Section heading */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(102,126,234,0.5);
    }

    /* Attack table */
    .attack-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }
    .attack-table th {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0 16px 8px;
        text-align: left;
    }
    .attack-table td {
        background: rgba(255,255,255,0.06);
        color: white;
        padding: 12px 16px;
        font-size: 0.9rem;
    }
    .attack-table tr td:first-child { border-radius: 10px 0 0 10px; }
    .attack-table tr td:last-child  { border-radius: 0 10px 10px 0; }

    /* Pill tag */
    .pill {
        background: rgba(102,126,234,0.25);
        color: #a78bfa;
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 36px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 8px 24px rgba(102,126,234,0.4) !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(102,126,234,0.55) !important;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        color: rgba(255,255,255,0.55);
        font-size: 0.8rem;
        margin-top: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Info box */
    .info-box {
        background: rgba(102,126,234,0.15);
        border: 1px solid rgba(102,126,234,0.35);
        border-radius: 12px;
        padding: 16px 20px;
        color: rgba(255,255,255,0.85);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Streamlit overrides */
    .stSelectbox > div > div,
    .stRadio > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    label { color: rgba(255,255,255,0.75) !important; font-size: 0.85rem !important; }
    .stSlider > div > div > div { background: #667eea !important; }
    h1,h2,h3,h4 { color: white !important; }
    p, li { color: rgba(255,255,255,0.8) !important; }
    .stSpinner > div { border-top-color: #667eea !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Lazy import attack core ────────────────────────────────────────────────
@st.cache_resource
def load_core():
    from transfer_attack_core import (
        ATTACKER_MODELS, VICTIM_MODELS,
        load_and_preprocess, compute_embedding,
        attack_loss, denormalize, build_attacker,
        pgd_attack, mi_fgsm, ti_fgsm, vmi_fgsm,
        EPSILON, NUM_ITER
    )
    return {
        'ATTACKER_MODELS': ATTACKER_MODELS,
        'load_and_preprocess': load_and_preprocess,
        'compute_embedding': compute_embedding,
        'attack_loss': attack_loss,
        'denormalize': denormalize,
        'build_attacker': build_attacker,
        'pgd_attack': pgd_attack,
        'mi_fgsm': mi_fgsm,
        'ti_fgsm': ti_fgsm,
        'vmi_fgsm': vmi_fgsm,
        'EPSILON': EPSILON,
        'NUM_ITER': NUM_ITER,
    }

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ FaceGuard</h1>
    <p>Interactive demonstration of adversarial transfer attacks on face verification systems.<br>
    Upload two faces, choose an attack, and see how AI models can be fooled.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Attack Configuration")
    st.markdown("---")

    attacker_model = st.selectbox(
        "Surrogate Model",
        ["Facenet512", "ArcFace", "GhostFaceNet", "VGG-Face"],
        help="The model used to generate the adversarial perturbation"
    )

    attack_name = st.selectbox(
        "Attack Method",
        ["VMI_FGSM", "MI_FGSM", "PGD", "TI_FGSM"],
        help="The adversarial attack algorithm to apply"
    )

    attack_type = st.radio(
        "Attack Goal",
        ["impersonation_attack", "dodging_attack"],
        format_func=lambda x: "🎭 Impersonation" if x == "impersonation_attack" else "🫥 Dodging"
    )

    st.markdown("---")
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                         help="Cosine similarity above this = same person")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:rgba(255,255,255,0.5); line-height:1.7'>
    <b style='color:rgba(255,255,255,0.8)'>🎭 Impersonation</b><br>
    Fool model into thinking Person A is Person B<br><br>
    <b style='color:rgba(255,255,255,0.8)'>🫥 Dodging</b><br>
    Fool model into not recognizing Person A<br><br>
    <b style='color:rgba(255,255,255,0.8)'>Threshold</b><br>
    Similarity above this = "same person"
    </div>
    """, unsafe_allow_html=True)

# ── Upload section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📁 Upload Face Images</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Source Image** — the face to be attacked")
    src_file = st.file_uploader("", type=["jpg","jpeg","png"], key="src", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Target Image** — the identity to impersonate (or same person for dodging)")
    tgt_file = st.file_uploader("", type=["jpg","jpeg","png"], key="tgt", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main logic ─────────────────────────────────────────────────────────────
if src_file and tgt_file:
    src_img = Image.open(src_file).convert('RGB')
    tgt_img = Image.open(tgt_file).convert('RGB')
    src_img.save("temp_src.jpg")
    tgt_img.save("temp_tgt.jpg")

    # Preview
    st.markdown('<div class="section-title">🖼️ Uploaded Images</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.image(src_img, use_container_width=True)
        st.markdown('<div class="img-caption">Source Face</div>', unsafe_allow_html=True)
    with p2:
        st.image(tgt_img, use_container_width=True)
        st.markdown('<div class="img-caption">Target Face</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button(f"⚡ Run {attack_name.replace('_',' ')} Attack")

    if run:
        with st.spinner("Generating adversarial example... please wait"):
            try:
                core = load_core()
                input_size = core['ATTACKER_MODELS'][attacker_model]

                src_arr = core['load_and_preprocess']("temp_src.jpg", input_size)
                tgt_arr = core['load_and_preprocess']("temp_tgt.jpg", input_size)
                src_tf  = tf.expand_dims(tf.constant(src_arr), 0)
                tgt_tf  = tf.expand_dims(tf.constant(tgt_arr), 0)

                model   = core['build_attacker'](attacker_model)
                tgt_emb = core['compute_embedding'](model, tgt_tf)
                src_emb = core['compute_embedding'](model, src_tf)
                clean_sim = float(tf.reduce_sum(src_emb * tgt_emb))

                # Run attack
                if attack_name == "VMI_FGSM":
                    adv = core['vmi_fgsm'](model, src_tf, tgt_emb, attack_type , n=5)
                elif attack_name == "MI_FGSM":
                    adv = core['mi_fgsm'](model, src_tf, tgt_emb, attack_type)
                elif attack_name == "PGD":
                    adv = core['pgd_attack'](model, src_tf, tgt_emb, attack_type)
                elif attack_name == "TI_FGSM":
                    adv = core['ti_fgsm'](model, src_tf, tgt_emb, attack_type)

                adv_emb   = core['compute_embedding'](model, adv)
                adv_sim   = float(tf.reduce_sum(adv_emb * tgt_emb))
                adv_img   = core['denormalize'](adv.numpy()[0])
                adv_pil   = Image.fromarray(adv_img)

                src_display = np.array(src_img.resize(input_size))
                adv_display = np.array(adv_pil)
                perturbation = np.abs(adv_display.astype(int) - src_display.astype(int)).astype(np.uint8)
                perturb_amp  = np.clip(perturbation * 10, 0, 255).astype(np.uint8)

                if attack_type == "impersonation_attack":
                    breach = adv_sim >= threshold
                else:
                    breach = adv_sim < threshold

                # ── Results header
                st.markdown("---")
                st.markdown('<div class="section-title">🔬 Attack Results</div>', unsafe_allow_html=True)

                # ── Images
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    st.image(src_img.resize(input_size), use_container_width=True)
                    st.markdown('<div class="img-caption">Original Image</div>', unsafe_allow_html=True)
                with ic2:
                    st.image(adv_pil, use_container_width=True)
                    st.markdown('<div class="img-caption">Adversarial Image</div>', unsafe_allow_html=True)
                with ic3:
                    st.image(perturb_amp, use_container_width=True)
                    st.markdown('<div class="img-caption">Perturbation (10x)</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                delta = adv_sim - clean_sim
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                delta_color = "#38ef7d" if (delta > 0 and attack_type == "impersonation_attack") or (delta < 0 and attack_type == "dodging_attack") else "#ffd200"

                with mc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Clean Similarity</div>
                        <div class="metric-value">{clean_sim:.4f}</div>
                        <div class="metric-delta" style="color:rgba(255,255,255,0.4)">Before attack</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Adversarial Similarity</div>
                        <div class="metric-value">{adv_sim:.4f}</div>
                        <div class="metric-delta" style="color:{delta_color}">{delta_str}</div>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Threshold</div>
                        <div class="metric-value">{threshold:.2f}</div>
                        <div class="metric-delta" style="color:rgba(255,255,255,0.4)">Decision boundary</div>
                    </div>""", unsafe_allow_html=True)
                with mc4:
                    badge = f'<span class="badge-success">✓ SUCCESS</span>' if breach else f'<span class="badge-fail">✗ FAILED</span>'
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Attack Result</div>
                        <div style="margin-top:12px">{badge}</div>
                        <div class="metric-delta" style="color:rgba(255,255,255,0.4);margin-top:10px">{'Breach' if breach else 'No breach'}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Chart
                fig, ax = plt.subplots(figsize=(7, 3.2))
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')

                bar_colors = ['#667eea', '#38ef7d' if breach else '#ffd200']
                bars = ax.bar(['Clean Similarity', 'Adversarial Similarity'],
                             [clean_sim, adv_sim],
                             color=bar_colors, width=0.45,
                             edgecolor='white', linewidth=0.5)
                ax.axhline(y=threshold, color='#f093fb', linestyle='--',
                          linewidth=1.8, label=f'Threshold ({threshold})', alpha=0.85)
                ax.set_ylim(-0.1, 1.05)
                ax.set_ylabel('Cosine Similarity', color='white', fontsize=11)
                ax.tick_params(colors='white', labelsize=10)
                ax.spines['bottom'].set_color('#444444')
                ax.spines['left'].set_color('#444444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                           f'{h:.4f}', ha='center', va='bottom',
                           color='white', fontsize=11, fontweight='bold')
                legend = ax.legend(facecolor='none', edgecolor='#444444',
                                  labelcolor='white', fontsize=10)
                st.pyplot(fig, transparent=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Explanation
                if attack_type == "impersonation_attack":
                    if breach:
                        msg = f"The attack <b>succeeded</b>. By adding imperceptible noise to the source image, the model's similarity score jumped from <b>{clean_sim:.3f}</b> to <b>{adv_sim:.3f}</b> — crossing the decision threshold of <b>{threshold}</b>. The face recognition system now incorrectly identifies the source person as the target identity."
                        color = "rgba(56,239,125,0.15)"
                        border = "rgba(56,239,125,0.4)"
                    else:
                        msg = f"The attack <b>failed</b>. Similarity changed from <b>{clean_sim:.3f}</b> to <b>{adv_sim:.3f}</b>, but did not cross the threshold of <b>{threshold}</b>. Try increasing epsilon or switching to VMI-FGSM."
                        color = "rgba(255,210,0,0.12)"
                        border = "rgba(255,210,0,0.4)"
                else:
                    if breach:
                        msg = f"The attack <b>succeeded</b>. The similarity score dropped from <b>{clean_sim:.3f}</b> to <b>{adv_sim:.3f}</b> — falling below the threshold of <b>{threshold}</b>. The face recognition system can no longer identify this person."
                        color = "rgba(56,239,125,0.15)"
                        border = "rgba(56,239,125,0.4)"
                    else:
                        msg = f"The attack <b>failed</b>. Similarity changed from <b>{clean_sim:.3f}</b> to <b>{adv_sim:.3f}</b>, but stayed above the threshold of <b>{threshold}</b>. Try a different attack method."
                        color = "rgba(255,210,0,0.12)"
                        border = "rgba(255,210,0,0.4)"

                st.markdown(f"""
                <div style="background:{color}; border:1px solid {border};
                     border-radius:14px; padding:18px 22px;
                     color:rgba(255,255,255,0.88); font-size:0.95rem; line-height:1.7">
                    💡 <b>What happened?</b><br>{msg}
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure both images contain clear, visible faces.")

else:
    # ── Landing info
    st.markdown("---")
    st.markdown('<div class="section-title">🧠 About the Attacks</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="attack-table">
    <tr>
        <th>Attack</th><th>Paper</th><th>Key Idea</th><th>Strength</th>
    </tr>
    <tr>
        <td><span class="pill">VMI-FGSM</span></td>
        <td>CVPR 2021</td>
        <td>Variance-tuned momentum — samples gradient neighbourhood to reduce variance</td>
        <td>⭐⭐⭐⭐⭐</td>
    </tr>
    <tr>
        <td><span class="pill">MI-FGSM</span></td>
        <td>CVPR 2018</td>
        <td>Momentum accumulation for stable gradient direction</td>
        <td>⭐⭐⭐⭐</td>
    </tr>
    <tr>
        <td><span class="pill">TI-FGSM</span></td>
        <td>CVPR 2019</td>
        <td>Translation-invariant perturbations via Gaussian kernel smoothing</td>
        <td>⭐⭐⭐</td>
    </tr>
    <tr>
        <td><span class="pill">PGD</span></td>
        <td>ICLR 2018</td>
        <td>Projected gradient descent with epsilon-ball constraint</td>
        <td>⭐⭐</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <ol style="color:rgba(255,255,255,0.85); line-height:2.2; font-size:0.95rem">
        <li>Upload a <b>source face image</b> — this is the face that will be perturbed</li>
        <li>Upload a <b>target face image</b> — the identity to impersonate (or same person for dodging)</li>
        <li>Choose your <b>attack method</b> and <b>goal</b> in the sidebar</li>
        <li>Adjust the <b>decision threshold</b> if needed</li>
        <li>Click <b>Run Attack</b> and observe the results</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.25); font-size:0.78rem; padding-bottom:20px">
    FaceGuard — Adversarial Face Verification Demo &nbsp;|&nbsp;
    For educational and research purposes only &nbsp;|&nbsp;
    Built with Streamlit + TensorFlow + DeepFace
</div>
""", unsafe_allow_html=True)