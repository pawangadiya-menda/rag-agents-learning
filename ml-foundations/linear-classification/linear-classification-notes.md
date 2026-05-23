# Linear Classification — Concepts, Math & Real Life Examples

> **PM Lens:** This post documents my journey learning ML fundamentals through a product management lens — connecting math to real business decisions.

---

## 🗺️ What We Cover

1. [Classification — The Umbrella](#1-classification--the-umbrella)
2. [Linear Classification — The Algorithm](#2-linear-classification--the-algorithm)
3. [The Math — wᵀx Explained](#3-the-math--wtx-explained)
4. [Training — How Weights Update](#4-training--how-weights-update)
5. [η (Eta) — The Learning Rate](#5-η-eta--the-learning-rate)
6. [Full End-to-End Example](#6-full-end-to-end-example)
7. [Linear Classifier vs Logistic Regression](#7-linear-classifier-vs-logistic-regression)
8. [Linear Regression vs Linear Classifier vs Logistic Regression](#8-linear-regression-vs-linear-classifier-vs-logistic-regression)
9. [sklearn Code](#9-sklearn-code)
10. [PM Lens — Real Life Use Cases](#10-pm-lens--real-life-use-cases)

---

## 1. Classification — The Umbrella

Classification is the **task** — given an input, predict which category it belongs to.

```
CLASSIFICATION (The Problem)
"Predict which category does this belong to?"
        │
        ├── Linear Classifier      ← covered in this post
        ├── Logistic Regression    ← covered in this post
        ├── Decision Tree
        ├── Random Forest
        ├── SVM
        └── Neural Network
```

**Real examples:**
- Will this B2B lead convert? → Yes / No
- Is this transaction fraudulent? → Fraud / Not Fraud
- Will this customer churn? → Churn / Stay

---

## 2. Linear Classification — The Algorithm

Linear classification solves the classification task by drawing a **straight line** (or flat hyperplane) to separate classes.

```
x₂ ↑
   |    ● ●  ← Converted leads
   |  ●
   |─────────── Decision Boundary (wᵀx = 0)
   |    ○ ○  ← Not converted
   |  ○
   └──────────────→ x₁
```

**The core constraint:** The boundary must be straight.

---

## 3. The Math — wᵀx Explained

### Score Formula

```
score = wᵀx = w₀·x₀ + w₁·x₁ + w₂·x₂

score > 0  →  predict +1 (Convert)
score ≤ 0  →  predict -1 (Don't Convert)
score = 0  →  Decision Boundary
```

### What is wᵀx?

`wᵀx` is just a **dot product** — three ways to write the same thing:

| Notation | Form |
|---|---|
| `wᵀx` | Matrix form |
| `w · x` | Dot product form |
| `Σ wᵢxᵢ` | Summation form |

### What is x₀?

`x₀ = 1` always. It exists so the **bias weight w₀** can be updated the same way as all other weights.

Without bias → boundary is forced through the origin (too restrictive).  
With bias → boundary can shift freely anywhere in feature space.

### Why Transpose?

`w` and `x` are both column vectors (n×1). To multiply them and get a **scalar**, we transpose `w` to (1×n):

```
wᵀ (1×n)  ×  x (n×1)  =  scalar (1×1) ✅
```

---

## 4. Training — How Weights Update

### The Perceptron Update Rule

```
w_new = w_old + η · y · x

Only applied when prediction is WRONG.
```

| Piece | Meaning |
|---|---|
| `w_old` | Current weights |
| `η` | Learning rate — size of correction step |
| `y` | True label of the misclassified point |
| `x` | Feature vector of that point |

### Why This Formula Works

**Case 1 — True y=+1, predicted -1 (score too negative):**
```
η · y · x = positive → weights GO UP → score increases → fixes prediction
```

**Case 2 — True y=-1, predicted +1 (score too positive):**
```
η · y · x = negative → weights GO DOWN → score decreases → fixes prediction
```

**Why multiply by x?**  
Not all features contributed equally to the wrong prediction. Multiplying by x ensures **bigger features get bigger corrections** — proportional fix.

---

## 5. η (Eta) — The Learning Rate

η is **how much you let one mistake change your mind.**

### Chai Making Analogy ☕

```
Making chai. Mom says "too sweet." How much do you reduce sugar?

η = 1.0  →  Remove ALL sugar  →  now not sweet at all  →  overcorrects ❌
η = 0.01 →  Remove tiny bit   →  barely changes        →  too slow ❌
η = 0.1  →  Remove 10%        →  steady improvement    →  just right ✅
```

### Visual Effect on Training

```
η too HIGH          η too LOW           η just right
Loss│/\/\/\         Loss│\              Loss│\
    │       \           │ \............     │ \____
    └──epochs           └──epochs           └──epochs
    Bounces forever     Never converges     Converges ✅
```

### Why η Exists

1. **Prevent overshooting** — one big jump can make things worse
2. **Handle noisy data** — don't overreact to outliers
3. **Balance speed vs stability** — too small = slow, too big = unstable

---

## 6. Full End-to-End Example

**Use Case:** B2B SaaS Lead Scoring (CRM company like Salesforce)

### Dataset

| Lead | x₁ Company Size | x₂ Visits/Month | Label y |
|---|---|---|---|
| L1 | 0.8 | 0.9 | +1 (Converted) |
| L2 | 0.6 | 0.7 | +1 (Converted) |
| L3 | 0.75 | 0.8 | +1 (Converted) |
| L4 | 0.2 | 0.3 | -1 (Not Converted) |
| L5 | 0.15 | 0.25 | -1 (Not Converted) |
| L6 | 0.3 | 0.2 | -1 (Not Converted) |

*Note: x₀ = 1 added to all leads (bias term)*

### Initial Setup
```
w = [0.0,  0.1,  0.1]
η = 0.1
```

### Epoch 1 Summary

```
L1: score=0.17  → predict +1  true +1  ✅  no update
L2: score=0.13  → predict +1  true +1  ✅  no update
L3: score=0.155 → predict +1  true +1  ✅  no update
L4: score=0.05  → predict +1  true -1  ❌  UPDATE
    η·y·x = 0.1×(-1)×[1,0.2,0.3] = [-0.1,-0.02,-0.03]
    w = [0.0-0.1, 0.1-0.02, 0.1-0.03] = [-0.1, 0.08, 0.07]
L5: score=-0.07 → predict -1  true -1  ✅  no update
L6: score=-0.06 → predict -1  true -1  ✅  no update

End Epoch 1: w=[-0.1, 0.08, 0.07]  |  5/6 correct
```

### Epoch 3 — Convergence

```
L1: score=+0.095 → +1 ✅
L2: score=+0.049 → +1 ✅
L3: score=+0.078 → +1 ✅
L4: score=-0.043 → -1 ✅
L5: score=-0.054 → -1 ✅
L6: score=-0.042 → -1 ✅

All 6/6 correct → STOP TRAINING ✅
Final w = [-0.1, 0.12, 0.11]
```

### Weight Journey

```
         w₀      w₁      w₂     Correct
Start:  [0.0,   0.1,   0.1 ]    3/6
Ep 1:   [-0.1,  0.08,  0.07]    5/6
Ep 2:   [-0.1,  0.12,  0.11]    4/6
Ep 3:   [-0.1,  0.12,  0.11]    6/6  ← converged
```

### Predicting New Leads

```
L7: Company=500, Visits=550 → x=[1, 0.5, 0.55]
score = (-0.1×1)+(0.12×0.5)+(0.11×0.55) = +0.0205 → CONVERT ✅

L8: Company=100, Visits=180 → x=[1, 0.1, 0.18]
score = (-0.1×1)+(0.12×0.1)+(0.11×0.18) = -0.068  → DON'T CONVERT ❌
```

---

## 7. Linear Classifier vs Logistic Regression

Both draw a **straight line boundary.** Key difference: **output type.**

| | Linear Classifier | Logistic Regression |
|---|---|---|
| **Output** | +1 or -1 | Probability 0.0–1.0 |
| **Answers** | "Will convert?" | "How likely to convert?" |
| **Ranking** | No | Yes |
| **Use when** | Binary action needed | Prioritization needed |

### The Sigmoid Function

Logistic Regression wraps the score through sigmoid:

```
σ(score) = 1 / (1 + e^(-score))

score = -4  →  2%  probability
score =  0  →  50% probability  (boundary)
score = +4  →  98% probability
```

### When to Use Which

**Use Linear Classifier when:**
- Email spam filter (move to spam or not)
- Fraud block (block transaction or allow)
- Manufacturing QC (pass or fail)
- Action is instant and binary

**Use Logistic Regression when:**
- Lead scoring (rank 200 leads, call top 20)
- Churn intervention (how much discount to offer)
- Credit risk (what interest rate to charge)
- You need priority order, not just yes/no

---

## 8. Linear Regression vs Linear Classifier vs Logistic Regression

```
LINEAR REGRESSION      →  predict a NUMBER
LINEAR CLASSIFIER      →  predict YES or NO
LOGISTIC REGRESSION    →  predict PROBABILITY (0 to 1)
```

### Same Data, Three Different Questions

```
Data: Company size, website visits

Q1. "How much revenue will this customer generate?"
    → Linear Regression → ₹4,50,000

Q2. "Will this customer convert?"
    → Linear Classifier → YES or NO

Q3. "How likely is this customer to convert?"
    → Logistic Regression → 78%
```

### The Math — All Three Start the Same

```
score = wᵀx  (same for all three)

Linear Regression:    output = wᵀx              (raw number)
Linear Classifier:    output = sign(wᵀx)         (+1 or -1)
Logistic Regression:  output = σ(wᵀx)            (0 to 1)
```

### Decision Guide

```
Output needed?
│
├── A NUMBER (revenue, price, time)
│   → LINEAR REGRESSION
│
└── A CATEGORY
    │
    ├── Need probability / ranking?
    │   → LOGISTIC REGRESSION
    │
    └── Pure binary, speed matters?
        → LINEAR CLASSIFIER
```

---

## 9. sklearn Code

```python
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Dataset
X = np.array([
    [800, 900],   # L1 - converted
    [600, 700],   # L2 - converted
    [750, 800],   # L3 - converted
    [200, 300],   # L4 - not converted
    [150, 250],   # L5 - not converted
    [300, 200],   # L6 - not converted
])
y = np.array([1, 1, 1, -1, -1, -1])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Linear Classifier (Perceptron) ──────────────────────
perceptron = Perceptron(eta0=0.1, max_iter=100)
perceptron.fit(X_scaled, y)
print("Perceptron prediction:", perceptron.predict(scaler.transform([[500, 600]])))
# Output: [1]  →  Will Convert

# ── Logistic Regression ─────────────────────────────────
log_reg = LogisticRegression()
log_reg.fit(X_scaled, y)
print("LogReg prediction:    ", log_reg.predict(scaler.transform([[500, 600]])))
print("LogReg probability:   ", log_reg.predict_proba(scaler.transform([[500, 600]])))
# Output: [1]  →  Will Convert
# Output: [[0.12, 0.88]]  →  88% chance of converting

# ── Linear Regression ───────────────────────────────────
# Predicting revenue (continuous output)
y_revenue = np.array([450, 380, 420, 120, 90, 150])  # in ₹thousands
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y_revenue)
print("Revenue forecast:     ₹", lin_reg.predict(scaler.transform([[500, 600]])))
# Output: ₹ [310.5]  →  ₹3,10,500 expected revenue
```

---

## 10. PM Lens — Real Life Use Cases

| Company | Problem | Algorithm | Output |
|---|---|---|---|
| **Salesforce** | Lead priority | Logistic Regression | 89% convert probability |
| **Razorpay** | Fraud blocking | Linear Classifier | Block / Allow |
| **Swiggy** | Delivery time | Linear Regression | 34 minutes |
| **Hotstar** | Churn risk | Logistic Regression | 74% churn → trigger offer |
| **HDFC** | Loan approval | Logistic Regression | 23% default risk |
| **Zerodha** | Credit limit | Linear Regression | ₹8,50,000 |
| **Nykaa** | Spam reviews | Linear Classifier | Spam / Not Spam |

### The Full Training Loop

```
Initialize w randomly
        ↓
FOR each epoch:
  FOR each data point (x, y):
    score   = wᵀx          ← dot product
    predict = sign(score)  ← +1 or -1
    if wrong:
      w = w + η · y · x   ← η controls step
                              y controls direction
                              x controls proportion
  if all correct → STOP
        ↓
Model trained → predict new leads
```

---

## Key Takeaways

1. **Classification** is the problem. Linear Classifier, Logistic Regression etc. are the solutions.
2. **wᵀx** is just a dot product — measures alignment between weights and features.
3. **x₀ = 1** always — lets bias weight fit into the same formula.
4. **η** controls step size of correction. Too high = bounces. Too low = slow. ~0.1 is good start.
5. **Iterations** give the model multiple passes to fix all mistakes.
6. **Linear Classifier** → yes/no. **Logistic Regression** → probability. **Linear Regression** → number.
7. In B2B SaaS, **Logistic Regression wins most of the time** — business decisions need probability, not just yes/no.

---

*📚 Part of my AI/ML learning series with a PM lens — connecting math to product decisions.*  
*Course: IIIT Hyderabad AI/ML Program*  
*Reference: Hands-On ML (Aurélien Géron), Chapters 2–3*
