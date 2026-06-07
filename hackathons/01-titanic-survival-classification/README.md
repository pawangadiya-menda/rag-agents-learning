# 🚢 Hackathon 1 — Titanic Survival Classification

> **IIIT Hyderabad × TalentSprint AI/ML Programme**  
> Binary classification using a VotingClassifier ensemble

---

## 🔗 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJ79SmkRGwS3rQtdNOJ7Q8lsc643i3oK#scrollTo=Xe5puM-Hrm1G)

---

## 📌 Problem Statement

Predict whether a Titanic passenger **survived (1)** or **did not survive (0)** based on demographic and ticket features.

- **Type:** Binary Classification  
- **Train set:** 891 rows  
- **Test set:** 418 rows  
- **Target variable:** `Survived`

---

## 🧠 Solution Walkthrough

### Exercise 1 — Data Exploration
Loaded `titanic.csv` and inspected shape, dtypes, and null counts.

Key findings:
| Column | Missing Values | Action |
|---|---|---|
| `Age` | 177 | Fill with random value in [mean−std, mean+std] |
| `Cabin` | 687 | Convert to binary `Has_Cabin` flag |
| `Embarked` | 2 | Fill with mode |

### Exercise 2 — Train / Validation Split
Split data **before any preprocessing** using `train_test_split` to prevent data leakage.

### Exercise 3 — Cleaning & Feature Engineering
| Feature | Transformation |
|---|---|
| `Age` | Mean ± std random imputation |
| `Cabin` | → `Has_Cabin` binary (1/0) |
| `Embarked` | Mode fill → label encoded |
| `Sex` | Label encoded (male=1, female=0) |
| `Fare` | Binned into categories |

### Exercise 4 — VotingClassifier Ensemble
```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
    ('svm', SVC())
], voting='hard')
```

### Exercise 5 — Test Set Preprocessing
Applied **identical transformations** using training statistics only — no leakage.

### Exercise 6 — Submission File
Generated `submission.csv` with PassengerId + Survived predictions.

---

## 📊 Model Performance

| Model | Validation Accuracy |
|---|---|
| Logistic Regression | 81.56% |
| Decision Tree | 81.56% |
| Random Forest | 80.45% |
| SVM | 81.01% |
| **🏆 VotingClassifier** | **~82%** |

---

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Google Colab`

---

## 💡 Key Learnings

1. **Ensemble = wisdom of crowds** — combining 4 models reduces variance even when individual accuracy is similar
2. **fit_transform vs transform** — fit only on training data; apply (transform) to val/test
3. **Data leakage** — always split first, then preprocess
4. **Feature engineering signal** — `Has_Cabin` flag outperformed raw cabin string encoding
5. **Cell order in Colab matters** — variable state is sequential; session restarts clear memory

---

*← [Back to Hackathons Overview](../README.md)*
