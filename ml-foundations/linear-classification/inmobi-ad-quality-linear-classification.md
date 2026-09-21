# Linear Classification @ InMobi — Ad Quality Scoring Engine
### A Product Manager's Case Study

> **Company:** InMobi — India's largest ad tech company. $300M revenue (FY2024).  
> **Business:** Programmatic mobile ad exchange connecting 40,000+ publisher apps with global advertisers.  
> **PM Problem:** Should we serve this ad impression or block it? Decide in <100ms.

---

## 🏢 InMobi Business Context

InMobi operates at the intersection of **two sides of a marketplace:**

```
ADVERTISER SIDE                    PUBLISHER SIDE
(Brands paying for reach)          (Apps earning from ads)
─────────────────────              ──────────────────────
Nike, Samsung, Amazon    ←→        Candy Crush, Hotstar,
Flipkart, Swiggy                   ShareChat, MX Player
Pay per impression/click           Earn per ad shown

         InMobi Exchange sits in the middle
         Runs 10B+ ad auctions per day
         Revenue = % of every transaction
```

**InMobi makes money only when:**
- Advertiser pays for an impression ✅
- Publisher earns from that impression ✅
- User doesn't have a bad experience ✅

All three must win simultaneously. That's the PM challenge.

---

## 🎯 The Core PM Problem

Every second, InMobi receives thousands of ad requests. Each is an auction — multiple advertisers bid to show their ad to a specific user on a specific app at a specific moment.

But not every ad should be served. Some are:

```
❌ Clickbait ("You won ₹10,000! Click now!")
❌ Malware disguised as ads
❌ Irrelevant (luxury car ad to a student)
❌ Low quality creatives (blurry, broken links)
❌ Brand unsafe (adult content on kids' app)
❌ Fraudulent (bot traffic, fake impressions)
```

Serving bad ads destroys **publisher trust** (users uninstall the app) and **advertiser ROI** (brand damage).

**The PM Question:**
> "How do we automatically decide — in under 100 milliseconds — whether an ad impression is worth serving or should be blocked?"

**Answer: Linear Classification — Ad Quality Scoring Engine**

---

## 💡 PM Hypothesis — What Signals Matter?

Before writing any code, a PM asks: **"What tells us an ad is high quality vs low quality?"**

```
HIGH QUALITY signals (+1 = Serve):
✅ High advertiser trust score (verified brand)
✅ High historical CTR (users actually click)
✅ Strong publisher-category match (gaming ad on gaming app)
✅ Low complaint rate (users don't report this ad)

LOW QUALITY signals (-1 = Block):
❌ Low advertiser trust score (new/unverified)
❌ Low or zero CTR history (nobody clicks)
❌ Poor category match (finance ad on kids app)
❌ High complaint rate (users report as annoying/misleading)
```

These become our **feature vector x.**

---

## 📦 Dataset — Historical Ad Decisions

InMobi's data team pulls 10 historical ad impressions with known outcomes:

| Ad ID | x₁ Trust Score | x₂ Hist. CTR% | x₃ Category Match | x₄ Complaint Rate% | Decision y |
|---|---|---|---|---|---|
| AD01 | 0.90 | 2.10 | 0.85 | 0.50 | **+1** Served |
| AD02 | 0.85 | 1.80 | 0.90 | 0.30 | **+1** Served |
| AD03 | 0.75 | 1.50 | 0.70 | 0.80 | **+1** Served |
| AD04 | 0.80 | 2.50 | 0.80 | 0.40 | **+1** Served |
| AD05 | 0.70 | 1.20 | 0.75 | 0.60 | **+1** Served |
| AD06 | 0.20 | 0.10 | 0.30 | 8.50 | **-1** Blocked |
| AD07 | 0.15 | 0.05 | 0.20 | 9.20 | **-1** Blocked |
| AD08 | 0.30 | 0.20 | 0.15 | 7.80 | **-1** Blocked |
| AD09 | 0.10 | 0.08 | 0.25 | 9.50 | **-1** Blocked |
| AD10 | 0.25 | 0.15 | 0.10 | 8.00 | **-1** Blocked |

**Feature normalization** (scale to 0–1 range):
```
x₁ = Trust Score         (already 0–1)
x₂ = CTR / 3.0           (max CTR ~3%)
x₃ = Category Match      (already 0–1)
x₄ = Complaint Rate / 10  (max ~10%)
x₀ = 1                   (bias term, always)
```

Normalized dataset:

| Ad ID | x₀ | x₁ Trust | x₂ CTR | x₃ Match | x₄ Complaint | y |
|---|---|---|---|---|---|---|
| AD01 | 1 | 0.90 | 0.70 | 0.85 | 0.05 | +1 |
| AD02 | 1 | 0.85 | 0.60 | 0.90 | 0.03 | +1 |
| AD03 | 1 | 0.75 | 0.50 | 0.70 | 0.08 | +1 |
| AD04 | 1 | 0.80 | 0.83 | 0.80 | 0.04 | +1 |
| AD05 | 1 | 0.70 | 0.40 | 0.75 | 0.06 | +1 |
| AD06 | 1 | 0.20 | 0.03 | 0.30 | 0.85 | -1 |
| AD07 | 1 | 0.15 | 0.02 | 0.20 | 0.92 | -1 |
| AD08 | 1 | 0.30 | 0.07 | 0.15 | 0.78 | -1 |
| AD09 | 1 | 0.10 | 0.03 | 0.25 | 0.95 | -1 |
| AD10 | 1 | 0.25 | 0.05 | 0.10 | 0.80 | -1 |

---

## ⚙️ Model Setup

```
w = [w₀,   w₁,    w₂,    w₃,    w₄  ]
  = [0.0,  0.2,   0.2,   0.2,  -0.2  ]

w₀ = bias
w₁ = weight for trust score      (+ve: high trust = serve)
w₂ = weight for CTR              (+ve: high CTR = serve)
w₃ = weight for category match   (+ve: good match = serve)
w₄ = weight for complaint rate   (-ve: high complaints = block)

η = 0.1
```

> **PM Note on initial weights:**
> We initialize w₄ as negative because we **know** complaints should reduce the ad quality score. This is domain knowledge injected upfront. A PM with InMobi's product context does this — you don't start blind.

---

## 🔄 Training — Epoch 1

**Score formula:**
```
score = w₀·x₀ + w₁·x₁ + w₂·x₂ + w₃·x₃ + w₄·x₄
```

---

**AD01 | x=[1, 0.90, 0.70, 0.85, 0.05] | y=+1**
```
score = (0.0×1)+(0.2×0.90)+(0.2×0.70)+(0.2×0.85)+(-0.2×0.05)
      = 0 + 0.18 + 0.14 + 0.17 - 0.01
      = 0.48 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD02 | x=[1, 0.85, 0.60, 0.90, 0.03] | y=+1**
```
score = (0.0×1)+(0.2×0.85)+(0.2×0.60)+(0.2×0.90)+(-0.2×0.03)
      = 0 + 0.17 + 0.12 + 0.18 - 0.006
      = 0.464 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD03 | x=[1, 0.75, 0.50, 0.70, 0.08] | y=+1**
```
score = (0.0×1)+(0.2×0.75)+(0.2×0.50)+(0.2×0.70)+(-0.2×0.08)
      = 0 + 0.15 + 0.10 + 0.14 - 0.016
      = 0.374 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD04 | x=[1, 0.80, 0.83, 0.80, 0.04] | y=+1**
```
score = (0.0×1)+(0.2×0.80)+(0.2×0.83)+(0.2×0.80)+(-0.2×0.04)
      = 0 + 0.16 + 0.166 + 0.16 - 0.008
      = 0.478 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD05 | x=[1, 0.70, 0.40, 0.75, 0.06] | y=+1**
```
score = (0.0×1)+(0.2×0.70)+(0.2×0.40)+(0.2×0.75)+(-0.2×0.06)
      = 0 + 0.14 + 0.08 + 0.15 - 0.012
      = 0.358 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD06 | x=[1, 0.20, 0.03, 0.30, 0.85] | y=-1**
```
score = (0.0×1)+(0.2×0.20)+(0.2×0.03)+(0.2×0.30)+(-0.2×0.85)
      = 0 + 0.04 + 0.006 + 0.06 - 0.17
      = -0.064 → predict -1  ✅ CORRECT

PM Insight: High complaint rate (0.85) correctly crushed the score.
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD07 | x=[1, 0.15, 0.02, 0.20, 0.92] | y=-1**
```
score = (0.0×1)+(0.2×0.15)+(0.2×0.02)+(0.2×0.20)+(-0.2×0.92)
      = 0 + 0.03 + 0.004 + 0.04 - 0.184
      = -0.11 → predict -1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD08 | x=[1, 0.30, 0.07, 0.15, 0.78] | y=-1**
```
score = (0.0×1)+(0.2×0.30)+(0.2×0.07)+(0.2×0.15)+(-0.2×0.78)
      = 0 + 0.06 + 0.014 + 0.03 - 0.156
      = -0.052 → predict -1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD09 | x=[1, 0.10, 0.03, 0.25, 0.95] | y=-1**
```
score = (0.0×1)+(0.2×0.10)+(0.2×0.03)+(0.2×0.25)+(-0.2×0.95)
      = 0 + 0.02 + 0.006 + 0.05 - 0.19
      = -0.114 → predict -1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

**AD10 | x=[1, 0.25, 0.05, 0.10, 0.80] | y=-1**
```
score = (0.0×1)+(0.2×0.25)+(0.2×0.05)+(0.2×0.10)+(-0.2×0.80)
      = 0 + 0.05 + 0.01 + 0.02 - 0.16
      = -0.08 → predict -1  ✅ CORRECT
      No update. w = [0.0, 0.2, 0.2, 0.2, -0.2]
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EPOCH 1 COMPLETE
w = [0.0, 0.2, 0.2, 0.2, -0.2]
Correct: 10/10 ✅ CONVERGED IN 1 EPOCH!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **PM Note:** Converged in 1 epoch because good ads and bad ads have very distinct signal patterns. Complaint rate alone almost perfectly separates them. In real InMobi systems with thousands of features and noisy data — it would take many more epochs.

---

## 🚀 Real Time Scoring — New Ad Requests

**Model trained. w = [0.0, 0.2, 0.2, 0.2, -0.2]. Decision needed in <100ms.**

---

**NEW AD A: Samsung Galaxy S25**
```
Trust Score: 0.95 | CTR: 2.4% → 0.80 | Match: 0.88 | Complaints: 0.3% → 0.03
x = [1, 0.95, 0.80, 0.88, 0.03]

score = (0.0×1)+(0.2×0.95)+(0.2×0.80)+(0.2×0.88)+(-0.2×0.03)
      = 0 + 0.19 + 0.16 + 0.176 - 0.006
      = +0.52 → ✅ SERVE

PM Read: Verified brand, high CTR, low complaints. Easy yes.
```

---

**NEW AD B: "Win ₹50,000 — Click Now!" Clickbait**
```
Trust Score: 0.12 | CTR: 0.08% → 0.03 | Match: 0.15 | Complaints: 9.1% → 0.91
x = [1, 0.12, 0.027, 0.15, 0.91]

score = (0.0×1)+(0.2×0.12)+(0.2×0.027)+(0.2×0.15)+(-0.2×0.91)
      = 0 + 0.024 + 0.005 + 0.03 - 0.182
      = -0.123 → ❌ BLOCK

PM Read: 9.1% complaint rate is the killer signal. Block immediately.
Protects MX Player's user experience and InMobi's publisher trust.
```

---

**NEW AD C: Flipkart Big Billion Days Sale**
```
Trust Score: 0.88 | CTR: 1.9% → 0.63 | Match: 0.72 | Complaints: 0.5% → 0.05
x = [1, 0.88, 0.63, 0.72, 0.05]

score = (0.0×1)+(0.2×0.88)+(0.2×0.63)+(0.2×0.72)+(-0.2×0.05)
      = 0 + 0.176 + 0.126 + 0.144 - 0.01
      = +0.436 → ✅ SERVE

PM Read: Flipkart sale on ShareChat — Bharat audience, strong match.
```

---

**NEW AD D: Loan App on Candy Crush**
```
Trust Score: 0.35 | CTR: 0.45% → 0.15 | Match: 0.10 | Complaints: 6.5% → 0.65
x = [1, 0.35, 0.15, 0.10, 0.65]

score = (0.0×1)+(0.2×0.35)+(0.2×0.15)+(0.2×0.10)+(-0.2×0.65)
      = 0 + 0.07 + 0.03 + 0.02 - 0.13
      = -0.01 → ❌ BLOCK

PM Read: Loan ads on Candy Crush = brand safety disaster.
Low match + high complaints. Protects publisher from user backlash.
```

---

## 📊 Decision Summary

```
Ad Request            Score    Decision   Key Reason
──────────────────────────────────────────────────────────────
Samsung Galaxy S25    +0.52    ✅ SERVE   High trust + high CTR
Flipkart BBD Sale     +0.44    ✅ SERVE   Trusted brand, solid match
Loan App Candy Crush  -0.01    ❌ BLOCK   Poor match + high complaints
Clickbait Win ₹50K    -0.12    ❌ BLOCK   9.1% complaint rate = killer
```

---

## 🔁 The Full Loop — PM Translation

```
ML Step                InMobi PM Translation
───────────────────    ──────────────────────────────────────────
Initialize weights     PM sets quality thresholds using
                       domain knowledge (complaints penalized)

Forward pass           Score every ad request in real time
score = wᵀx           <100ms decision per impression

Wrong prediction       A bad ad got served → publisher complaint
                       A good ad got blocked → lost revenue

Weight update          Model recalibrates quality thresholds
w = w + η·y·x         Based on outcome signals (complaints,
                       CTR data, publisher feedback)

η = 0.1               Don't overhaul scoring after one bad ad.
                       Update gradually. Stable marketplace.

Epoch                  Daily model retraining cycle
                       New complaint data → updated weights

Convergence            Model consistently blocks bad ads,
                       serves good ads → publisher NPS up,
                       advertiser ROI up, revenue grows
```

---

## 💡 PM Decisions Driven by This Model

### Decision 1 — Advertiser Onboarding Policy
```
Model shows: Trust Score weight = positive, high impact
             New advertisers (trust=0.1–0.3) almost always blocked

PM Action: New advertisers must complete verification
           before first impression is served.
           Build "Advertiser Trust Score" as a product feature.
```

### Decision 2 — Publisher Category Enforcement
```
Model shows: Category Match weight = positive
             Finance ads on kids apps = always blocked

PM Action: Build "Category Exclusion Lists" for publishers.
           Let Candy Crush say "no finance, no gambling."
           Automated enforcement via model.
```

### Decision 3 — Complaint Rate SLA
```
Model shows: Complaint Rate = STRONGEST blocking signal

PM Action: Set SLA: any advertiser crossing 3% complaint rate
           gets auto-suspended. Account team gets alert.
           Build this as an automated enforcement feature.
```

### Decision 4 — η for Model Retraining
```
Daily retraining (η=0.1):
  + Catches new scam ad patterns faster
  - Risk of overfitting to one bad day's data

Weekly retraining (η=0.05):
  + Stable scoring for advertisers
  - Slow to catch new fraud patterns

PM Decision: Daily retraining with η=0.05
             Smaller step, stable, but still responsive.
```

---

## 🐍 sklearn Code — InMobi Ad Quality Scorer

```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Historical ad decisions (training data) ─────────────
X = np.array([
    # [trust_score, hist_ctr%, category_match, complaint_rate%]
    [0.90, 2.10, 0.85, 0.50],  # AD01 - Served
    [0.85, 1.80, 0.90, 0.30],  # AD02 - Served
    [0.75, 1.50, 0.70, 0.80],  # AD03 - Served
    [0.80, 2.50, 0.80, 0.40],  # AD04 - Served
    [0.70, 1.20, 0.75, 0.60],  # AD05 - Served
    [0.20, 0.10, 0.30, 8.50],  # AD06 - Blocked
    [0.15, 0.05, 0.20, 9.20],  # AD07 - Blocked
    [0.30, 0.20, 0.15, 7.80],  # AD08 - Blocked
    [0.10, 0.08, 0.25, 9.50],  # AD09 - Blocked
    [0.25, 0.15, 0.10, 8.00],  # AD10 - Blocked
])

y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
# +1 = SERVE, -1 = BLOCK

# ── Normalize ───────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ───────────────────────────────────────────────
model = Perceptron(eta0=0.1, max_iter=1000, random_state=42)
model.fit(X_scaled, y)
print("Training Accuracy:", accuracy_score(y, model.predict(X_scaled)))

# ── Real time scoring — new ad requests ─────────────────
new_ads = np.array([
    [0.95, 2.40, 0.88, 0.30],  # Samsung Galaxy S25
    [0.12, 0.08, 0.15, 9.10],  # Clickbait Win 50K
    [0.88, 1.90, 0.72, 0.50],  # Flipkart BBD Sale
    [0.35, 0.45, 0.10, 6.50],  # Loan App Candy Crush
])

ad_names = [
    "Samsung Galaxy S25",
    "Clickbait Win 50K",
    "Flipkart BBD Sale",
    "Loan App Candy Crush"
]

new_scaled = scaler.transform(new_ads)
predictions = model.predict(new_scaled)
scores = model.decision_function(new_scaled)

print("\n── Real Time Ad Quality Decisions ──")
print(f"{'Ad':<25} {'Score':>8}  Decision")
print("─" * 48)
for name, score, pred in zip(ad_names, scores, predictions):
    decision = "✅ SERVE" if pred == 1 else "❌ BLOCK"
    print(f"{name:<25} {score:>8.3f}  {decision}")

# ── Feature importance ───────────────────────────────────
print("\n── Feature Weight Analysis ──")
features = ["Trust Score", "Hist CTR", "Category Match", "Complaint Rate"]
weights = model.coef_[0]
for feat, w in sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑ SERVE signal" if w > 0 else "↓ BLOCK signal"
    print(f"{feat:<20} weight={w:>7.3f}  {direction}")
```

---

## 📈 Business Impact

```
WITHOUT Ad Quality Scoring:
Manual review → 100 ads/day max
10B+ impressions/day → impossible to scale
Bad ads leak through → publisher churn
Publisher churn → less inventory → advertisers leave

WITH Linear Classification:
Automated → scores 10B+ impressions/day in <100ms
Bad ads blocked → publisher NPS improves
Publisher NPS up → more apps join exchange
More apps → more advertiser demand
→ Revenue growing 35–40% YoY (FY2025 target)

InMobi recognized as "Strong Performer" in
Forrester Wave: Sell-Side Platforms Q4 2024
Top marks in: Ad Quality Controls, Audience Targeting
```

---

## 🔑 Key Takeaways for PMs

```
1. LINEAR CLASSIFICATION = REAL TIME DECISIONING AT SCALE
   10B+ decisions/day that no human team can make.
   Model makes them in <100ms, consistently.

2. WEIGHTS = PRODUCT POLICY IN MATH FORM
   Complaint weight negative = your publisher protection policy.
   Change the weight → change the policy.
   Audit the weights → audit your product decisions.

3. TRAINING DATA = PAST PRODUCT DECISIONS
   Your model is only as good as your historical decisions.
   "Clean training data" is a PM responsibility, not just ML team's.

4. η = CHANGE MANAGEMENT SPEED
   High η → model changes fast → advertisers see erratic scoring.
   Low η  → model changes slow → new fraud patterns slip through.
   PM owns this tradeoff. It's a product policy decision.

5. DECISION BOUNDARY = YOUR QUALITY BAR
   score = 0 is where you're 50/50 on serve/block.
   Stricter threshold (+0.1) → fewer bad ads, but also less revenue.
   Looser threshold (-0.1) → more revenue, more brand risk.
   PM owns this threshold. It's a business decision.
```

---

*📚 Part of my AI/ML learning series with a PM lens — connecting math to product decisions.*
*Course: IIIT Hyderabad AI/ML Program*
*Target roles: Technical PM at Salesforce, SAP, ServiceNow, Adobe, Sprinklr*
*Author: Pawan Kumar Gadiya | Senior PM @ Accenture | ISB MBA 2023*
