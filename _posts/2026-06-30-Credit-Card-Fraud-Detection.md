# Credit Card Fraud Detection Project

An end-to-end machine learning project evaluating various anomaly detection models on highly imbalanced credit card transaction data.

---

## 📌 Project Overview & Problem Statement

In real-world financial systems, fraudulent transactions account for a minuscule fraction of overall volume. However, leaving them undetected results in massive financial losses, while over-aggressive models disrupt legitimate user behavior by increasing customer friction.

The primary objective of this project was to build and evaluate multiple modeling paradigms to determine the most robust architecture for isolating fraudulent behavior. Because the dataset is heavily skewed, **Precision-Recall Area Under the Curve (PR-AUC)** was utilized as our primary evaluation metric, optimizing for the best operational balance between minimizing false negatives (Recall) and minimizing false positives (Precision).

---

## 📊 Data Profile

The dataset contains real credit card transactions made by European cardholders. Due to confidentiality reasons, the original features have been transformed using Principal Component Analysis (PCA) into 28 orthogonal linear components (`V1` through `V28`), alongside the raw transaction `Amount` and `Time`.

* **Total Transactions:** 284,807
* **Legitimate Transactions:** 284,315 (99.83%)
* **Fraudulent Transactions:** 492 (0.17%)
* **Imbalance Ratio:** ~578 to 1

---

## 🛠️ Evaluated Model Architectures

Several distinct modeling scenarios were implemented across three fundamental training paradigms:

1. **Supervised Linear Baseline:** Logistic Regression utilizing balanced class weights to mathematically inflate the penalty of missing a minority class.
2. **Supervised Tree Ensembles:** Tree-based structures (`Random Forest` and `XGBoost`) capable of drawing sharp, non-linear boundaries. Heavy class imbalances were managed using sample-level bootstrap weightings (`balanced_subsample`) and loss-gradient rescaling (`scale_pos_weight`).
3. **Unsupervised Anomaly Detection:** Models trained exclusively on normal transactions (`y == 0`) to learn a tight profile of legitimate behavior, treating fraud strictly as a statistical or structural outlier. This included a geometric **One-Class SVM** and a deep learning **PyTorch Autoencoder** utilizing reconstruction error (MSE) as an anomaly metric.

---

## 🏆 Final Model Performance Summary

The models below are ranked in descending order by their performance on the holdout test set. 

| Scenario | Model Architecture | Learning Type | Scale Required? | PR-AUC Score |
| :---: | :--- | :--- | :---: | :---: |
| **7** | **Cross-Validated XGBoost (Optimized)** | Supervised | No | **0.865028** |
| **2** | **Base Random Forest (Balanced Subsample)** | Supervised | No | **0.864048** |
| **4** | **Base XGBoost (scale_pos_weight)** | Supervised | No | **0.829893** |
| **6** | **Cross-Validated Random Forest (Optimized)** | Supervised | No | **0.808603** |
| **1** | **Logistic Regression (Balanced)** | Supervised | Yes | **0.763894** |
| **3** | **One-Class SVM (Anomaly Detection)** | Unsupervised | Yes | **0.322874** |
| **5** | **Autoencoder (PyTorch Unsupervised)** | Unsupervised | Yes | **0.248103** |

---

## 🔑 Core Engineering Takeaways

* **Supervised Models Outperformed Unsupervised models:** Having explicit visibility into historical fraud labels allowed models like XGBoost and Random Forest to carve out hyper-precise decision regions within the abstract PCA feature space.
* **The Anomaly Detection Barrier:** The One-Class SVM and Autoencoder struggled (PR-AUC $< 0.33$) because smart fraud often successfully mimics normal transaction coordinates. Furthermore, since the data was already compressed via PCA, there was minimal residual non-linear latent structure left for the deep neural network layers to compress and elegantly reconstruct.
* **Hyperparameter & Regularization Trade-offs:** Stratified 3-Fold Cross-Validation successfully pushed XGBoost to our peak performance (**0.8650**). For the Random Forest, the unconstrained baseline slightly outperformed the CV variant on this specific holdout partition; this indicates that the unconstrained trees effectively exploited deep, clean boundary lines unique to this pre-processed dataset.
