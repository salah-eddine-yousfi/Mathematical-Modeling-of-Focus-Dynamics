# Mathematical Modeling of Focus Dynamics
> **Written and edited by:** Salah Eddine YOUSFI  
> **Date:** 05/02/2026


Do not hesitate to contact me through my LinkedIn profile  


[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white&style=for-the-badge)](https://www.linkedin.com/in/salah-eddine-yousfi-b2532a375/)

---
## 🎥 Demo Video

[![Watch the demo](https://img.shields.io/badge/▶%20Watch-Demo-blue?style=for-the-badge)](https://github.com/salah-eddine-yousfi/Mathematical-Modeling-of-Focus-Dynamics/blob/5b2680bf7d189f35ae5c5d59daacfe562ecd7cf7/Mathematical%20Modeling%20Of%20Focus%20Dynamics%20Video.mp4)

# 🧠 From Real Images to a Credible Estimation of Human Concentration

This project explores a simple but challenging question:

> **Can human concentration be approximated from real images, without cheating on data or interpretation?**

The goal is **not** to measure concentration perfectly,  
but to propose a **coherent, interpretable and realistic methodology** that connects
visual perception to a cognitive state over time.

---

## 📸 1. Real Data, Fully Controlled

The project is built on a **self-collected dataset**:

- ≈ **2600 images**
- 📸 captured **by myself**
- under **real-world conditions**

I deliberately introduced variability in:
- fingernails,
- clothing,
- lighting conditions,
- backgrounds.

👉 **No synthetic or artificial data**  
👉 Only real, imperfect, real-life images.

---

## 📦 2. Dataset Designed to Avoid an “Easy” (Cheating) Model

High accuracy in vision can be misleading if the dataset is poorly split.
To avoid this, the dataset was designed to be **strict and honest**:

- private data,
- ❌ **no image-by-image split**  
  (which often shows almost identical images in train/val/test),
- ✅ **group split by fingernail**.

This guarantees that:
- the same fingernail **never appears** in train, validation and test,
- there is **no data leakage**,
- the model cannot rely on memorization.

👉 The model is **forced to generalize**.

---

## 👁️ 3. Learning to See Before Judging

The system does **not** start with a binary decision
("concentrated / not concentrated").

Instead, the neural network first learns **four visual classes**:

- `focused_writing`
- `focused_reading`
- `not_phone`
- `not_activity`

👉 This produces a **rich and realistic visual perception**  
👉 before any cognitive simplification.

---

## 🧠 4. Visual Model (Perception Only)

The neural network is used **only for visual perception**:

- **Architecture**: ResNet-18
- **Optimizer**: AdamW
- **Transfer learning**: ImageNet pretrained weights

Overfitting is controlled using:
- data augmentation (train only),
- weight decay,
- dropout,
- best model selection via validation loss.

### 📈 Performance on Unseen Data

- **test accuracy = 87.65%**
- strict test split by fingernail
- new lighting, clothing and backgrounds

👉 A **realistic performance**, not artificially inflated.

---

## 🧠 5. From Visual Classes to Cognitive States

Once visual perception is obtained, classes are grouped conceptually:

- **1️⃣ = concentrated**
  - focused_writing
  - focused_reading
- **0️⃣ = not concentrated**
  - not_phone
  - not_activity

This step bridges **vision** and **cognition**.

However, a key issue appears immediately.

---

## ❓ 6. Why Counting 0️⃣ / 1️⃣ Does Not Work

Human concentration is:
- not instantaneous,
- not frame-by-frame,
- not well described by isolated decisions.

Examples:

0 0 0 0 0 0 0 0 (1) 0 0 0 0 0 0 0 0
                  → real concentration or noise?

1 1 1 1 1 1 1 1 (0) 1 1 1 1 1 1 1 1                  
                  → real loss of focus or a brief perturbation?


👉 **Counting binary outputs has no cognitive meaning**.

---

## 📐 7. Core Idea: Mathematical Evidence Accumulation (Not a Learned Model)

To address this, the project uses **mathematical formulas**,  
**not a trained model**, inspired by cognitive science.

Reference:
> **Normative Evidence Accumulation in Unpredictable Environments**  
> Glaze et al., 2015

This framework describes how a rational agent should:
- **accumulate evidence over time**
- instead of deciding from instantaneous observations
- in noisy and unstable environments.

### Important clarification

- ❌ No neural network here
- ❌ No learning
- ❌ No backpropagation
- ❌ No fitted weights

✔ Only **explicit mathematical equations**  
✔ Fully **interpretable parameters**

---

## 🔢 How Evidence Accumulation Is Applied Here

From the binary visual outputs (1 / 0):

- each observation contributes **positive or negative evidence**,
- evidence is **integrated over time**,
- isolated contradictions are down-weighted,
- decisions depend on a **continuous internal state**, not a single frame.

Interpretation:
- several **1️⃣ in a row** → concentration installs progressively
- a single **0️⃣** → treated as noise
- several **0️⃣ in a row** → gradual exit from concentration

An intentional asymmetry is introduced:
- ⬆️ entering concentration is slow,
- ⬇️ leaving concentration is faster.

---

## ⚠️ Parameter Choice (Scientific Honesty)

All parameters:
- are **chosen manually**,
- inspired by the literature,
- used as **pragmatic approximations**.

This is necessary because:
- concentration is a **latent cognitive state**,
- it cannot be measured exactly,
- images provide only **indirect evidence**.

---

## 📈 8. Making Concentration Visible

The mathematical accumulation produces interpretable signals:

1. 📈 **Concentration curve** (continuous, between 0 and 1)
2. 🍩 **Global concentration percentage**
   - computed from the **surface above the threshold**
   - not from counting frames
3. 📊 **Time spent per cognitive activity**
4. 📊 **Activity dominance curves**
   - explain *why* concentration rises or falls

👉 The result is not just a score,  
👉 but an **explainable temporal behavior**.

---

## ⚠️ 9. Scope and Limitations

This approach remains an **approximation**.

Limitations:
- one subject,
- limited data volume,
- non-universal behavior.

The model:
- does **not** provide a universal truth,
- does **not** generalize automatically to all people or contexts.

👉 It provides a **context-dependent, interpretable estimate**.

---

## 🎯 10. True Objective of the Project

The objective is **not** to measure concentration perfectly.

It is to demonstrate how to:
- start from real visual data,
- avoid common dataset pitfalls,
- stabilize noisy predictions over time,
- make a complex cognitive state observable.

This framework is **adaptable**:
- remote work monitoring,
- online learning platforms,
- other cognitive monitoring contexts,

provided that:
- new data is collected,
- parameters are re-adjusted.

---

### 🧩 Key Takeaway

- Neural networks → **visual perception**
- Mathematics → **cognitive temporal modeling**
- No hidden learning
- Full interpretability

This project proposes a **methodology**, not a universal model.

---


Please feel free to reach out to me through my LinkedIn Profile if you have any further questions or would like to discuss this study in more detail.



[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white&style=for-the-badge)](https://www.linkedin.com/in/salah-eddine-yousfi-b2532a375/)
