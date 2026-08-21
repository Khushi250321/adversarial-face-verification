# 🛡️ FaceGuard — Adversarial Face Verification Demo

An interactive web application demonstrating transfer-based adversarial attacks on face verification systems. Upload two face images, select an attack method, and see how invisible perturbations can fool state-of-the-art face recognition models.

**Live Demo:** Run locally with `streamlit run app.py`

---

## What is this?

Face verification systems decide if two face images belong to the same person using cosine similarity scores. FaceGuard demonstrates how adversarial attacks can manipulate these scores by adding imperceptible noise to face images.

Two attack goals:
- **Impersonation** — fool the model into thinking Person A is Person B
- **Dodging** — fool the model into failing to recognize Person A

---

## Attacks Implemented

| Attack | Paper | Key Idea |
|--------|-------|----------|
| VMI-FGSM | CVPR 2021 | Variance-tuned momentum — samples gradient neighbourhood |
| MI-FGSM | CVPR 2018 | Momentum accumulation for stable gradient direction |
| TI-FGSM | CVPR 2019 | Translation-invariant perturbations via Gaussian smoothing |
| PGD | ICLR 2018 | Projected gradient descent with epsilon-ball constraint |

---

## Features

- Upload any two face images
- Choose attack method and goal from sidebar
- View original vs adversarial image side by side
- Perturbation map (amplified 10x for visibility)
- Real-time similarity scores before and after attack
- Visual bar chart with decision threshold
- Plain English explanation of what happened

---

## Tech Stack

- Python 3.11
- TensorFlow 2.13
- DeepFace
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

## Installation

```bash
git clone https://github.com/Khushi250321/adversarial-face-verification.git
cd adversarial-face-verification

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open `localhost:8501` in your browser.

---

## How it Works

1. Source face image is preprocessed and passed through the surrogate model
2. Adversarial perturbation is computed using the selected attack algorithm
3. Perturbation is added to the source image within an epsilon budget
4. The adversarial image is evaluated against the target embedding
5. Cosine similarity is compared against the decision threshold

---

## Results

VMI-FGSM achieved **40.74% overall breach rate** on official evaluation, outperforming all baseline attacks:

| Attack | Breach Rate |
|--------|-------------|
| VMI-FGSM | 40.74% |
| SI-NI-FGSM | 33.14% |
| MI-FGSM | 27.27% |
| MI-ADMIX-DI-TI | 25.38% |
| TI-FGSM | 21.59% |
| PGD | 17.61% |

---

## Ethical Note

This project is for educational and research purposes only. It demonstrates AI vulnerabilities to motivate better defenses. Do not use for unauthorized access or harm.

---

## License

MIT License