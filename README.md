# Fair Hiring Model: Bias Mitigation in AI

## 📌 Project Overview
This project aims to develop a **plug-and-play fairness module** that integrates into any AI workflow to continuously monitor and mitigate bias. It supports various data types (**tabular, images, text**) and operates in **binary/multiclass classification** and **regression** settings. 

The key innovation is the use of **adversarial training with a Gradient Reversal Layer (GRL)** to **remove biased information** from AI decisions while maintaining predictive accuracy.

## 🎯 Objectives
- ✅ **Ensure fairness** by minimizing the influence of sensitive attributes (e.g., gender, race) on hiring decisions.
- ✅ **Maintain model performance** while debiasing the learned representations.
- ✅ **Generalize to different datasets** even if they have different numbers of features.

## 🚀 How It Works
The model consists of:
1. **A Shared Feature Extractor** → Learns representations of candidate profiles.
2. **A Primary Predictor** → Predicts hiring suitability based on extracted features.
3. **An Adversarial Network** → Tries to predict sensitive attributes (e.g., gender). The **Gradient Reversal Layer (GRL)** ensures that the shared representation removes bias by making the adversary’s task harder.

```mermaid
graph TD
    A[Input Data (Candidate Features)] -->|Shared Representation| B(Feature Extractor)
    B -->|Fair Features| C[Hiring Decision Model]
    B -->|Reversed Gradient| D(Adversary - Sensitive Attribute Predictor)
    C -->|Predictions| E[Hiring Decision (Fair)]
    D -->|Predicts Gender?| F[Bias Signal for Backpropagation]
```

## 📊 Evaluating Fairness and Performance
We compute the following metrics:

### 1️⃣ **Hiring Decision Performance**
- **Accuracy (Primary Task)**: Measures how well the model predicts hiring suitability.
- **F1 Score (Primary Task)**: Evaluates the balance between precision and recall.

### 2️⃣ **Bias Detection Performance**
- **Adversary Accuracy (Sensitive Attribute Prediction)**:
  - 🔼 **High** → The adversary can predict gender, meaning bias is present.
  - 🔽 **Low** → The adversary struggles, meaning the model is fair.
- **Adversary F1 Score (Sensitive Attribute Prediction)**:
  - Measures how well gender is predicted.

### 🛠 **Desired Trade-Off**
| Scenario | Interpretation |
|----------|---------------|
| **High primary accuracy + High adversary accuracy** | 🚨 Model is biased; hiring decision leaks sensitive information. |
| **High primary accuracy + Low adversary accuracy** | ✅ Model is fair; hiring decision does not depend on sensitive attributes. |
| **Low primary accuracy + Low adversary accuracy** | 🤔 Model might be underfitting or fairness regularization is too strong. |
| **Low primary accuracy + High adversary accuracy** | ❌ Model is learning biases but failing at hiring decisions. |

## 🛠 How to Use
1. **Install dependencies:**
   ```bash
   pip install torch numpy scikit-learn
   ```
2. **Train the Model:**
   ```python
   model = FairHiringModel(input_dim=20, lambda_adv=1.0)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   ```
3. **Adjust Fairness During Training:**
   ```python
   model.lambda_adv = 0.5  # Weaker debiasing mid-training
   ```
4. **Monitor Performance & Fairness:**
   ```python
   primary_acc = accuracy_score(y_true, primary_preds_bin)
   sensitive_acc = accuracy_score(s_true, sensitive_preds_bin)
   print("Hiring Accuracy:", primary_acc, "Bias Detection Accuracy:", sensitive_acc)
   ```

---