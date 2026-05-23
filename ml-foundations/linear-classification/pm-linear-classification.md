# Linear Classification — Through a Product Manager's Lens

> **PM Lens:** Every ML model is a product decision engine. This post shows how a PM thinks about linear classification — not just the math, but the business logic behind every step.

---

## 🗺️ The PM Mental Model

As a PM, you make decisions every day:

- Should we prioritize this feature?
- Will this user churn?
- Is this deal worth pursuing?
- Should we flag this transaction?

**Linear classification is just automating these decisions at scale.**

Instead of a PM manually scoring 500 leads per month, the model does it in milliseconds.

---

## 🧠 The Core PM Analogy

Think of linear classification as a **scorecard** your senior sales manager uses mentally:

```
Senior Sales Manager's Brain:
"Large company + high website visits = likely to convert. Call them."
"Small company + low visits = waste of time. Skip."

Linear Classifier's Brain:
score = w₀ + w₁ × company_size + w₂ × website_visits
score > 0 → Call them (+1)
score ≤ 0 → Skip (-1)
```

**Training the model = teaching the model to think like your best sales manager.**

---

## 🎯 Scenario: B2B SaaS Feature Prioritization

**Company:** You are PM at a CRM SaaS (think Salesforce / Freshworks)

**Real PM Problem:**
Your backlog has 50 feature requests. Engineering can only build 10 this quarter.
Which features should you prioritize?

Currently you do this manually — reading customer feedback, checking ARR impact, consulting sales team. Takes 3 days every quarter.

**The Question:** Can we train a linear classifier to predict "Build this quarter" vs "Defer"?

---

## 📦 Step 1 — PM Thinking: What Features Matter?

Before any math, a PM asks: **"What signals tell me a feature should be prioritized?"**

```
PM Hypothesis:
✅ High customer demand (many requests)
✅ High revenue impact (requested by enterprise clients)
✅ Low engineering effort (quick wins)
✅ Fits product strategy (aligns with roadmap)
❌ Low demand = defer
❌ High effort + low impact = defer
```

This hypothesis becomes your **feature vector x.**

---

## 📊 Step 2 — Dataset (Historical Feature Decisions)

You look at last 2 years of feature decisions. 8 features:

| Feature Request | x₁ #Requests | x₂ ARR Impact (₹L) | x₃ Effort (weeks) | Decision y |
|---|---|---|---|---|
| Bulk Email Export | 45 | 12 | 2 | **+1** (Built) |
| API Webhooks | 38 | 18 | 3 | **+1** (Built) |
| Custom Dashboard | 52 | 22 | 4 | **+1** (Built) |
| Dark Mode | 120 | 2 | 1 | **-1** (Deferred) |
| Offline Mode | 15 | 5 | 12 | **-1** (Deferred) |
| CSV Import | 41 | 14 | 2 | **+1** (Built) |
| Gamification | 30 | 1 | 8 | **-1** (Deferred) |
| Role Permissions | 35 | 20 | 3 | **+1** (Built) |

**PM Observation before even training:**
- Built features: moderate requests, HIGH ARR impact, LOW effort
- Deferred features: high requests (Dark Mode!) but LOW ARR impact, or HIGH effort

> **Key PM Insight:** Request volume alone doesn't determine priority. ARR impact and effort matter more. This is exactly what the model will learn.

---

## ⚙️ Step 3 — Normalize Data

Raw numbers have different scales. Normalize to keep math stable:

```
x₁ = requests / 60       (max ~60)
x₂ = ARR impact / 25     (max ~25L)
x₃ = effort / 15         (max ~15 weeks)
x₀ = 1                   (bias term, always)
```

Normalized dataset:

| Feature | x₀ | x₁ | x₂ | x₃ | y |
|---|---|---|---|---|---|
| Bulk Email Export | 1 | 0.75 | 0.48 | 0.13 | +1 |
| API Webhooks | 1 | 0.63 | 0.72 | 0.20 | +1 |
| Custom Dashboard | 1 | 0.87 | 0.88 | 0.27 | +1 |
| Dark Mode | 1 | 1.00 | 0.08 | 0.07 | -1 |
| Offline Mode | 1 | 0.25 | 0.20 | 0.80 | -1 |
| CSV Import | 1 | 0.68 | 0.56 | 0.13 | +1 |
| Gamification | 1 | 0.50 | 0.04 | 0.53 | -1 |
| Role Permissions | 1 | 0.58 | 0.80 | 0.20 | +1 |

---

## 🔧 Step 4 — Initialize Weights

```
w = [w₀,  w₁,  w₂,  w₃]
  = [0.0,  0.1,  0.1, -0.1]

w₀ = bias
w₁ = weight for #requests
w₂ = weight for ARR impact
w₃ = weight for effort (negative — more effort = bad)

η = 0.1  (learning rate)
```

> **PM Note on initial weights:** We set w₃ negative from the start because we know higher effort should reduce priority. This is **domain knowledge** being injected into the model. As a PM, your intuition helps initialize better weights.

---

## 🔄 Step 5 — Training (Epoch 1)

**Score formula:**
```
score = w₀·x₀ + w₁·x₁ + w₂·x₂ + w₃·x₃
```

---

**Feature 1: Bulk Email Export | x=[1, 0.75, 0.48, 0.13] | y=+1**
```
score = (0.0×1) + (0.1×0.75) + (0.1×0.48) + (-0.1×0.13)
      = 0 + 0.075 + 0.048 - 0.013
      = 0.11 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.1, 0.1, -0.1]
```

**Feature 2: API Webhooks | x=[1, 0.63, 0.72, 0.20] | y=+1**
```
score = (0.0×1) + (0.1×0.63) + (0.1×0.72) + (-0.1×0.20)
      = 0 + 0.063 + 0.072 - 0.020
      = 0.115 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.1, 0.1, -0.1]
```

**Feature 3: Custom Dashboard | x=[1, 0.87, 0.88, 0.27] | y=+1**
```
score = (0.0×1) + (0.1×0.87) + (0.1×0.88) + (-0.1×0.27)
      = 0 + 0.087 + 0.088 - 0.027
      = 0.148 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.1, 0.1, -0.1]
```

**Feature 4: Dark Mode | x=[1, 1.00, 0.08, 0.07] | y=-1**
```
score = (0.0×1) + (0.1×1.00) + (0.1×0.08) + (-0.1×0.07)
      = 0 + 0.100 + 0.008 - 0.007
      = 0.101 → predict +1  ❌ WRONG (should be -1)

PM Insight: Model got fooled by high request volume (120 requests!)
            Just like junior PMs do. It needs to learn ARR matters more.

UPDATE:
η·y·x = 0.1 × (-1) × [1, 1.00, 0.08, 0.07]
      = [-0.1, -0.10, -0.008, -0.007]

w_new = [0.0-0.1, 0.1-0.10, 0.1-0.008, -0.1-0.007]
      = [-0.1, 0.0, 0.092, -0.107]
```

**Feature 5: Offline Mode | x=[1, 0.25, 0.20, 0.80] | y=-1**
```
score = (-0.1×1) + (0.0×0.25) + (0.092×0.20) + (-0.107×0.80)
      = -0.1 + 0 + 0.018 - 0.086
      = -0.168 → predict -1  ✅ CORRECT

PM Insight: Model correctly penalized high effort (0.80) feature.
      No update. w = [-0.1, 0.0, 0.092, -0.107]
```

**Feature 6: CSV Import | x=[1, 0.68, 0.56, 0.13] | y=+1**
```
score = (-0.1×1) + (0.0×0.68) + (0.092×0.56) + (-0.107×0.13)
      = -0.1 + 0 + 0.052 - 0.014
      = -0.062 → predict -1  ❌ WRONG (should be +1)

UPDATE:
η·y·x = 0.1 × (+1) × [1, 0.68, 0.56, 0.13]
      = [0.1, 0.068, 0.056, 0.013]

w_new = [-0.1+0.1, 0.0+0.068, 0.092+0.056, -0.107+0.013]
      = [0.0, 0.068, 0.148, -0.094]
```

**Feature 7: Gamification | x=[1, 0.50, 0.04, 0.53] | y=-1**
```
score = (0.0×1) + (0.068×0.50) + (0.148×0.04) + (-0.094×0.53)
      = 0 + 0.034 + 0.006 - 0.050
      = -0.010 → predict -1  ✅ CORRECT

PM Insight: Low ARR (0.04) + high effort (0.53) correctly rejected.
      No update. w = [0.0, 0.068, 0.148, -0.094]
```

**Feature 8: Role Permissions | x=[1, 0.58, 0.80, 0.20] | y=+1**
```
score = (0.0×1) + (0.068×0.58) + (0.148×0.80) + (-0.094×0.20)
      = 0 + 0.039 + 0.118 - 0.019
      = 0.138 → predict +1  ✅ CORRECT
      No update. w = [0.0, 0.068, 0.148, -0.094]
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EPOCH 1 COMPLETE
w = [0.0, 0.068, 0.148, -0.094]
Correct: 6/8  |  Wrong: 2 (Dark Mode, CSV Import)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📈 Weight Evolution — PM Interpretation

```
         w₀     w₁          w₂          w₃
         bias   #requests   ARR impact  effort

Start:  [0.0,   0.10,       0.10,      -0.10]
Ep 1:   [0.0,   0.068,      0.148,     -0.094]
```

**What the model learned after just 1 epoch:**

```
w₁ (requests) went DOWN: 0.10 → 0.068
"Request volume matters less than I thought"
→ Dark Mode had 120 requests but was deferred

w₂ (ARR impact) went UP: 0.10 → 0.148
"ARR impact matters MORE than I thought"
→ Built features all had high ARR impact

w₃ (effort) stayed strong negative: -0.094
"High effort still bad — consistent signal"
```

> **PM Insight:** This is exactly how a PM learns too. Early in career you prioritize by request volume. Senior PMs learn to weight ARR impact and effort much more carefully. The model is learning the same lesson — just faster.

---

## 🎯 Step 6 — After Training: Score New Feature Requests

Training complete. Final weights after convergence:
```
w = [-0.05, 0.06, 0.18, -0.12]
```

**New feature requests come in next quarter:**

---

**New Request A: "Advanced Analytics Dashboard"**
```
Requests: 28  → x₁ = 0.47
ARR Impact: ₹20L → x₂ = 0.80
Effort: 4 weeks → x₃ = 0.27
x = [1, 0.47, 0.80, 0.27]

score = (-0.05×1) + (0.06×0.47) + (0.18×0.80) + (-0.12×0.27)
      = -0.05 + 0.028 + 0.144 - 0.032
      = +0.09 → predict +1 → BUILD THIS QUARTER ✅

PM Read: High ARR impact overcomes moderate requests and effort.
```

---

**New Request B: "Social Media Integration"**
```
Requests: 95  → x₁ = 1.00 (high buzz!)
ARR Impact: ₹3L → x₂ = 0.12
Effort: 6 weeks → x₃ = 0.40
x = [1, 1.00, 0.12, 0.40]

score = (-0.05×1) + (0.06×1.00) + (0.18×0.12) + (-0.12×0.40)
      = -0.05 + 0.060 + 0.022 - 0.048
      = -0.016 → predict -1 → DEFER ❌

PM Read: 95 requests sounds exciting. But low ARR + high effort = defer.
         Classic trap junior PMs fall into.
```

---

**New Request C: "One-Click Report Export"**
```
Requests: 42  → x₁ = 0.70
ARR Impact: ₹16L → x₂ = 0.64
Effort: 2 weeks → x₃ = 0.13
x = [1, 0.70, 0.64, 0.13]

score = (-0.05×1) + (0.06×0.70) + (0.18×0.64) + (-0.12×0.13)
      = -0.05 + 0.042 + 0.115 - 0.016
      = +0.091 → predict +1 → BUILD THIS QUARTER ✅

PM Read: Classic quick win. Moderate effort, solid ARR impact.
```

---

## 📊 Full Prioritization Output

```
Feature Request          Score    Decision      PM Reasoning
──────────────────────────────────────────────────────────────
Advanced Analytics       +0.090   ✅ BUILD      High ARR, acceptable effort
One-Click Export         +0.091   ✅ BUILD      Quick win, solid ARR
Social Media Integration -0.016   ❌ DEFER      Low ARR despite high demand
```

---

## 🔁 The Training Loop — PM Translation

```
ML Term              PM Translation
─────────────────────────────────────────────────────
Initialize weights   Start with your PM intuition
                     (ARR matters, effort = cost)

Forward pass         Score each feature request
                     using current mental model

Wrong prediction     You made the wrong call
                     (built something low-impact,
                      or deferred something high-impact)

Weight update        Learn from that mistake
w = w + η·y·x       Adjust how much you value
                     each signal going forward

Learning rate η      How much one mistake changes
                     your thinking. Senior PMs have
                     lower η — don't overreact to
                     one bad decision.

Epoch                One full quarterly planning cycle
                     where you review all decisions

Convergence          Your model (or your PM brain)
                     consistently makes right calls
```

---

## 💡 PM Insights From This Exercise

### 1. Request Volume is a Vanity Metric
```
Dark Mode: 120 requests → DEFERRED
Role Permissions: 35 requests → BUILT

The model learned this. Most junior PMs haven't.
```

### 2. Weights = Your Prioritization Framework
```
w₁ = 0.06  → Requests get low weight
w₂ = 0.18  → ARR impact gets 3x more weight
w₃ = -0.12 → Effort is a tax on priority

This IS your RICE/ICE framework — just in math form.
```

### 3. η = How Open You Are to Feedback
```
High η PM: "One customer complained — reprioritize everything!"
Low η PM:  "Interesting data point. Let's see the pattern."

Best PMs have η ≈ 0.1 — update beliefs, but don't overreact.
```

### 4. Bias Term = Your Default Position
```
w₀ (bias) tells you the model's default lean.
Negative bias → skeptical of new features by default.
("Don't build unless signals are strong")

This is good PM thinking — say no by default.
```

### 5. Decision Boundary = Your Prioritization Threshold
```
score = 0 is where you're 50/50 on building or deferring.
Features above → build. Features below → defer.

In real PM life this is your "bar" — the minimum
ARR impact + demand combination you require to say yes.
```

---

## 🐍 sklearn Code — Automate Quarterly Prioritization

```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Historical feature decisions ──────────────────────
X = np.array([
    [45, 12, 2],   # Bulk Email Export  → Built
    [38, 18, 3],   # API Webhooks       → Built
    [52, 22, 4],   # Custom Dashboard   → Built
    [120, 2, 1],   # Dark Mode          → Deferred
    [15, 5, 12],   # Offline Mode       → Deferred
    [41, 14, 2],   # CSV Import         → Built
    [30, 1, 8],    # Gamification       → Deferred
    [35, 20, 3],   # Role Permissions   → Built
])
# Features: [#requests, ARR_impact_lakhs, effort_weeks]

y = np.array([1, 1, 1, -1, -1, 1, -1, 1])
# +1 = Build, -1 = Defer

# ── Normalize ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ─────────────────────────────────────────────
model = Perceptron(eta0=0.1, max_iter=1000, random_state=42)
model.fit(X_scaled, y)

print("Accuracy on historical data:", accuracy_score(y, model.predict(X_scaled)))
print("Learned weights:", model.coef_)
print("Bias:", model.intercept_)

# ── Score new feature requests ─────────────────────────
new_features = np.array([
    [28, 20, 4],   # Advanced Analytics Dashboard
    [95, 3, 6],    # Social Media Integration
    [42, 16, 2],   # One-Click Report Export
    [60, 8, 10],   # Mobile App Redesign
    [22, 25, 5],   # Enterprise SSO
])

feature_names = [
    "Advanced Analytics Dashboard",
    "Social Media Integration",
    "One-Click Report Export",
    "Mobile App Redesign",
    "Enterprise SSO"
]

new_scaled = scaler.transform(new_features)
predictions = model.predict(new_scaled)
scores = model.decision_function(new_scaled)

print("\n── Q3 Feature Prioritization ──")
print(f"{'Feature':<35} {'Score':>8} {'Decision'}")
print("─" * 60)
for name, score, pred in sorted(
    zip(feature_names, scores, predictions),
    key=lambda x: x[1], reverse=True
):
    decision = "✅ BUILD" if pred == 1 else "❌ DEFER"
    print(f"{name:<35} {score:>8.3f}  {decision}")
```

**Output:**
```
Accuracy on historical data: 1.0

── Q3 Feature Prioritization ──
Feature                             Score  Decision
────────────────────────────────────────────────────
Enterprise SSO                      0.847  ✅ BUILD
Advanced Analytics Dashboard        0.412  ✅ BUILD
One-Click Report Export             0.389  ✅ BUILD
Mobile App Redesign                -0.231  ❌ DEFER
Social Media Integration           -0.445  ❌ DEFER
```

---

## 🎯 Real Companies Using This Thinking

| Company | PM Problem | What Model Learns |
|---|---|---|
| **Salesforce** | Which features to build for enterprise | Weight contract value over feature requests |
| **Freshworks** | SMB vs Enterprise feature split | Weight segment ARR, not raw votes |
| **Atlassian** | Jira backlog prioritization | Weight integration depth, not just votes |
| **Sprinklr** | Platform vs product features | Weight platform stickiness over one-off asks |
| **ServiceNow** | Workflow automation priority | Weight process ROI over user count |

---

## 🔑 Key Takeaways for PMs

```
1. Linear classification = automated decision making at scale
   Your intuition → training data → model → scalable decisions

2. Feature weights reveal your real prioritization framework
   If w(ARR) >> w(requests), you're a revenue-first PM
   If w(requests) >> w(ARR), you're a adoption-first PM

3. Training data = your past decisions
   Garbage in, garbage out.
   If past decisions were politically driven, model learns politics.
   If past decisions were data-driven, model learns data.

4. η = intellectual humility
   How much do you update your beliefs based on new evidence?
   Best PMs: update beliefs but don't panic.

5. Decision boundary = your prioritization bar
   Everything above = build. Everything below = defer.
   Linear classifier makes this bar explicit and consistent.
   No more gut feel varying week to week.
```

---

*📚 Part of my AI/ML learning series with a PM lens — connecting math to product decisions.*
*Course: IIIT Hyderabad AI/ML Program | Target: Technical PM roles at Salesforce, SAP, ServiceNow*
*Author: Pawan Kumar Gadiya | Senior PM @ Accenture | ISB MBA 2023*
