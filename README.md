\# Adversarial Face Verification Dashboard



A Streamlit-based interactive dashboard to demonstrate transfer-based adversarial attacks on face verification systems.



Built as an extension of my Summer 2026 internship at \*\*DRDO SAG Lab\*\*, where I implemented and evaluated VMI-FGSM on CNN-based face recognition models.



\---



\## What is this project?



Face verification systems compare two face images and decide if they belong to the same person using cosine similarity scores. This project demonstrates how adversarial attacks can fool these systems by adding invisible noise to face images.



Two types of attacks are demonstrated:

\- \*\*Impersonation\*\* — make the model think Person A is Person B

\- \*\*Dodging\*\* — make the model fail to recognize Person A



\---



\## Attacks Implemented



| Attack | Paper | Description |

|--------|-------|-------------|

| VMI-FGSM | CVPR 2021 | Variance-Tuned Momentum Iterative FGSM — strongest transfer attack |

| MI-FGSM | CVPR 2018 | Momentum Iterative FGSM |

| PGD | ICLR 2018 | Projected Gradient Descent |

| TI-FGSM | CVPR 2019 | Translation Invariant FGSM |



\---



\## Features



\- Upload any two face images

\- Select attack method and parameters

\- View original vs adversarial image side by side

\- See perturbation map (amplified 10x)

\- Real-time similarity scores before and after attack

\- Attack success/failure result with explanation



\---



\## Tech Stack



\- Python 3.11

\- TensorFlow 2.13

\- DeepFace

\- Streamlit

\- NumPy

\- Matplotlib

\- Pillow



\---



\## Models Supported



\*\*Attacker (Surrogate) Models:\*\*

\- FaceNet512

\- ArcFace

\- GhostFaceNet

\- VGG-Face



\---



\## Installation



```bash

\# Clone the repository

git clone https://github.com/Khushi250321/adversarial-face-verification.git

cd adversarial-face-verification



\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements.txt



\# Run the app

streamlit run app.py

```



\---



\## Usage



1\. Open the app at `localhost:8501`

2\. Upload a \*\*source face image\*\* (the face to be attacked)

3\. Upload a \*\*target face image\*\* (the identity to impersonate)

4\. Select attack configuration in the sidebar

5\. Click \*\*Run Attack\*\*

6\. View results — original vs adversarial image, similarity scores, attack outcome



\---



\## Internship Context



The attack implementations (VMI-FGSM, MI-FGSM, PGD, TI-FGSM) were developed during my Summer 2026 internship at \*\*DRDO SAG Lab\*\* under mentor Sanchit Gupta, as part of research on adversarial robustness of face recognition systems.



This Streamlit dashboard was independently built after the internship to demonstrate the attacks interactively for portfolio purposes.



\---



\## Results (from DRDO evaluation)



VMI-FGSM achieved an overall breach rate of \*\*40.74%\*\* on the official evaluation pipeline, outperforming all baseline attacks:



| Attack | Breach Rate |

|--------|-------------|

| VMI-FGSM | 40.74% |

| SI-NI-FGSM | 33.14% |

| MI-FGSM | 27.27% |

| MI-ADMIX-DI-TI | 25.38% |

| TI-FGSM | 21.59% |

| PGD | 17.61% |



\---



\## Ethical Note



This project is for educational and research purposes only. All experiments use consented public face images. The attacks demonstrated here highlight vulnerabilities in AI systems to motivate better defenses — not to enable unauthorized access.



\---



\## Author



\*\*Khushi\*\* — B.Tech Information Technology, IGDTUW  

Summer Internship 2026 — DRDO SAG Lab

