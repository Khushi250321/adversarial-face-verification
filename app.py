import streamlit as st
import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import io

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, r'C:\transferattacktInterns\core')
from transfer_attack_core import (
    ATTACKER_MODELS, VICTIM_MODELS,
    load_and_preprocess, compute_embedding,
    attack_loss, denormalize, build_attacker,
    pgd_attack, mi_fgsm, ti_fgsm, vmi_fgsm, EPSILON, NUM_ITER
)

st.set_page_config(
    page_title="Adversarial Face Verification",
    page_icon="🔐",
    layout="wide"
)

st.title("Adversarial Face Verification Dashboard")
st.markdown("**DRDO Internship Project — Khushi (IGDTUW)**")
st.markdown("Evaluate transfer-based adversarial attacks on face verification systems.")
st.markdown("---")

# Sidebar
st.sidebar.header("Attack Configuration")

attacker_model = st.sidebar.selectbox(
    "Attacker Model (Surrogate)",
    list(ATTACKER_MODELS.keys())
)

attack_name = st.sidebar.selectbox(
    "Attack Method",
    ["VMI_FGSM", "MI_FGSM", "PGD", "TI_FGSM"]
)

attack_type = st.sidebar.radio(
    "Attack Goal",
    ["impersonation_attack", "dodging_attack"],
    format_func=lambda x: "Impersonation" if x == "impersonation_attack" else "Dodging"
)

epsilon = st.sidebar.slider("Epsilon (Perturbation Budget)", 0.01, 0.1, 0.062, 0.005)
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**What is this?**")
st.sidebar.markdown("""
- **Impersonation**: Make model think person A is person B
- **Dodging**: Make model fail to recognize person A
- **Epsilon**: How much noise is added (smaller = less visible)
- **Threshold**: Similarity score above this = match
""")

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Image")
    src_file = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="src")

with col2:
    st.subheader("Target Image")
    tgt_file = st.file_uploader("Upload Target Face", type=["jpg", "jpeg", "png"], key="tgt")

if src_file and tgt_file:
    st.markdown("---")

    # Save uploaded files temporarily
    src_img = Image.open(src_file).convert('RGB')
    tgt_img = Image.open(tgt_file).convert('RGB')

    src_path = "temp_src.jpg"
    tgt_path = "temp_tgt.jpg"
    src_img.save(src_path)
    tgt_img.save(tgt_path)

    input_size = ATTACKER_MODELS[attacker_model]

    if st.button("Run Attack", type="primary"):
        with st.spinner("Running attack... this may take a minute..."):
            try:
                # Load and preprocess
                src_arr = load_and_preprocess(src_path, input_size)
                tgt_arr = load_and_preprocess(tgt_path, input_size)
                src_tf = tf.expand_dims(tf.constant(src_arr), 0)
                tgt_tf = tf.expand_dims(tf.constant(tgt_arr), 0)

                # Build model
                model = build_attacker(attacker_model)
                tgt_emb = compute_embedding(model, tgt_tf)
                src_emb = compute_embedding(model, src_tf)

                # Clean similarity
                clean_sim = float(tf.reduce_sum(src_emb * tgt_emb))

                # Run attack
                if attack_name == "VMI_FGSM":
                    adv = vmi_fgsm(model, src_tf, tgt_emb, attack_type)
                elif attack_name == "MI_FGSM":
                    adv = mi_fgsm(model, src_tf, tgt_emb, attack_type)
                elif attack_name == "PGD":
                    adv = pgd_attack(model, src_tf, tgt_emb, attack_type)
                elif attack_name == "TI_FGSM":
                    adv = ti_fgsm(model, src_tf, tgt_emb, attack_type)

                # Adversarial similarity
                adv_emb = compute_embedding(model, adv)
                adv_sim = float(tf.reduce_sum(adv_emb * tgt_emb))

                # Get adversarial image
                adv_img = denormalize(adv.numpy()[0])
                adv_pil = Image.fromarray(adv_img)

                # Perturbation
                src_display = np.array(src_img.resize(input_size))
                adv_display = np.array(adv_pil)
                perturbation = np.abs(adv_display.astype(int) - src_display.astype(int)).astype(np.uint8)
                perturb_amplified = np.clip(perturbation * 10, 0, 255).astype(np.uint8)

                # Display results
                st.subheader("Results")

                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(src_img.resize(input_size), caption="Original Image", use_container_width=True)
                with img_col2:
                    st.image(adv_pil, caption="Adversarial Image", use_container_width=True)
                with img_col3:
                    st.image(perturb_amplified, caption="Perturbation (10x amplified)", use_container_width=True)

                st.markdown("---")
                st.subheader("Similarity Scores")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Clean Similarity", f"{clean_sim:.4f}")
                with metric_col2:
                    st.metric("Adversarial Similarity", f"{adv_sim:.4f}", 
                             delta=f"{adv_sim - clean_sim:.4f}")
                with metric_col3:
                    if attack_type == "impersonation_attack":
                        breach = adv_sim >= threshold
                    else:
                        breach = adv_sim < threshold
                    st.metric("Attack Result", 
                             "SUCCESS" if breach else "FAILED",
                             delta="Breach" if breach else "No Breach")

                # Similarity bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(["Clean Similarity", "Adversarial Similarity"], 
                             [clean_sim, adv_sim],
                             color=["#2196F3", "#F44336" if breach else "#4CAF50"])
                ax.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
                ax.set_ylim(-1, 1)
                ax.set_ylabel("Cosine Similarity")
                ax.set_title(f"{attack_name} — {attack_type.replace('_', ' ').title()}")
                ax.legend()
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("What happened?")
                if attack_type == "impersonation_attack":
                    if breach:
                        st.success(f"The attack successfully made the model think the source person is the target person. Similarity rose from {clean_sim:.3f} to {adv_sim:.3f}, crossing the threshold of {threshold}.")
                    else:
                        st.warning(f"The attack failed. Similarity changed from {clean_sim:.3f} to {adv_sim:.3f}, but did not cross the threshold of {threshold}.")
                else:
                    if breach:
                        st.success(f"The attack successfully made the model fail to recognize the person. Similarity dropped from {clean_sim:.3f} to {adv_sim:.3f}, below the threshold of {threshold}.")
                    else:
                        st.warning(f"The attack failed. Similarity changed from {clean_sim:.3f} to {adv_sim:.3f}, but stayed above the threshold of {threshold}.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure both images are valid face images.")

    else:
        # Show preview
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.image(src_img, caption="Source Image", width=200)
        with prev_col2:
            st.image(tgt_img, caption="Target Image", width=200)
        st.info("Configure your attack in the sidebar and click 'Run Attack'")

else:
    st.info("Please upload both a source face image and a target face image to begin.")
    st.markdown("""
    ### How to use:
    1. Upload a **source image** (the face to be attacked)
    2. Upload a **target image** (the identity to impersonate, or same person for dodging)
    3. Select your **attack configuration** in the sidebar
    4. Click **Run Attack**
    
    ### About the attacks:
    | Attack | Description |
    |--------|-------------|
    | VMI-FGSM | Variance-Tuned Momentum — strongest transfer attack (CVPR 2021) |
    | MI-FGSM | Momentum Iterative — stable gradient direction |
    | PGD | Projected Gradient Descent — basic iterative |
    | TI-FGSM | Translation Invariant — diverse transformations |
    """)

st.markdown("---")
st.caption("Adversarial Face Verification Dashboard | DRDO Internship Project | Khushi (IGDTUW) | 2026")