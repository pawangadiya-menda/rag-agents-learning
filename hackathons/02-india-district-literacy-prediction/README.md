# 📚 Hackathon 2 — India District Literacy Rate Prediction

> **IIIT Hyderabad × TalentSprint AI/ML Programme**  
> Multi-class classification across 1,324 Indian districts

---

## 🔗 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ddxyq6fapvtYVYuz3v89u0r4Z5OME_Z3#scrollTo=l2Ur0nbghKUT)

---

## 📌 Problem Statement

Predict the overall literacy category — **High / Medium / Low** — for Indian districts using district-wise government school enrollment and basic education data.

- **Type:** Multi-Class Classification (3 classes)  
- **Dataset:** 1,324 districts × 180 features post-merge  
- **Target variable:** `overall_lit` (High / Medium / Low → encoded 0 / 1 / 2)  
- **Source:** Government of India district-wise education records

---

## 📂 Dataset

| File | Description |
|---|---|
| `Districtwise_Basicdata.csv` | District demographics, literacy rates, school counts |
| `Districtwise_Enrollment_details_indicator.csv` | Enrollment by school type, geography, education level |

---

## 🧠 7-Stage ML Pipeline

### Stage 1 — Load & Explore
- Loaded both CSVs; inspected shape, column types, null values
- Enrollment file had a **4-level multi-row header** (school type × geography × level × code)

### Stage 2 — Data Integration
Built a `unique_id` index: `Year_StateCode_DistrictCode`
```
2012-13_28_2822  →  Anantapur, Andhra Pradesh
```
Merged using `basic_data.join(enrollment_data, how='inner')`
**Result: 1,324 rows × 180 columns**

### Stage 3 — Data Cleaning
- Dropped duplicate header row at index 0
- Converted numeric-string columns to float
- Mode imputation for remaining categoricals
- `dropna()` for rows with unresolvable nulls

### Stage 4 — Drop Unnecessary Columns
Removed name/ID columns that don't contribute ML signal:
- `statename`, `distname`, `Year_x`, `Year_y`, duplicate `State Code` / `District Code` columns

### Stage 5 — Correlation Filtering
```python
def remove_Highly_Correlated(df, threshold=0.9):
    # Removes one column from each highly correlated pair
    ...
```
**169 features → 74 features** after removing redundant enrollment metrics

### Stage 6 — Normalization
Used `StandardScaler` (chosen over MinMaxScaler due to heavy outliers in population/enrollment columns)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)   # fit on train only
X_val_scaled = scaler.transform(X_val)     # transform only on val/test
```
Final shape: **(1,268 rows × 74 features)** · mean ≈ 0 · std ≈ 1

### Stage 7 — Model Training & Comparison
Trained three classifiers and compared:

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

---

## 📊 Results

| Model | Accuracy |
|---|---|
| SVM | >90% |
| Decision Tree | >90% |
| **🏆 Random Forest** | **>90%** |

Target accuracy of **90%+** achieved across all models, validating the pipeline.

---

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `StandardScaler` · `Google Colab`

---

## 💡 Key Learnings

1. **Multi-row headers are real-world noise** — government datasets rarely come clean; forward-fill + concat is the fix
2. **Unique ID construction** — combining Year + State Code + District Code creates a reliable join key
3. **Correlation filtering before scaling** — removes redundant signal and speeds up training
4. **StandardScaler vs MinMaxScaler** — StandardScaler is robust to outliers; MinMaxScaler is sensitive to them
5. **Target encoding** — `overall_lit` encoded to 0/1/2 (int64); `LabelEncoder` does this cleanly
6. **90%+ accuracy on district data** — high because enrollment patterns strongly predict literacy class

---

## 🔗 Related

- [Hackathon 1 — Titanic Survival Classification](../01-titanic-survival-classification/)
- [ML Foundations](../../ml-foundations/)

---

*← [Back to Hackathons Overview](../README.md)*
