# ⚙️ KNN Intrusive Task Detector

This project demonstrates a from-scratch implementation of the **K-Nearest Neighbors (KNN)** algorithm to classify whether a **CPU task is intrusive or not** based on its characteristics. 

---

## 📁 Dataset: `cpu_tasks.csv`

The dataset contains **1000 CPU tasks** with the following features:

| Column       | Description                                                  |
|--------------|--------------------------------------------------------------|
| `CPU_Usage`  | CPU usage percentage of the task (0 to 100)                 |
| `Priority`   | Task priority on a scale from 1 (low) to 10 (high)          |
| `Order`      | Execution order in the queue (1 = early, 20 = late)         |
| `Label`      | `intrusive` or `non-intrusive` — ground truth classification|


---

## 🔍 How KNN Works in This Project
1. For each test task, calculate the **Euclidean distance** to all training tasks.
2. Identify the **k closest tasks**.
3. Predict the label (`intrusive` or `non-intrusive`) based on **majority vote**.

---

## 🧪 Evaluation Metrics
- **Accuracy** is computed as:
  ```python
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(actual_labels, predicted_labels)
  ```

- **Confusion Matrix** is plotted using:
  ```python
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  ```

This helps visualize how well the model distinguishes between intrusive and non-intrusive tasks.

---

## 📈 Output Example
A sample confusion matrix for KNN classification:

|                | Predicted Intrusive | Predicted Non-Intrusive |
|----------------|---------------------|--------------------------|
| **Actual Intrusive**     | TP (True Positives)      | FN (False Negatives)         |
| **Actual Non-Intrusive** | FP (False Positives)     | TN (True Negatives)          |

This helps in evaluating **false alarms vs missed detections**.

---

## 🚀 How to Run
1. Generate the dataset (or use the existing `cpu_tasks.csv`)
2. Run the KNN classification script
3. Evaluate predictions and visualize results

---

## 📚 Dependencies
- Python 3.x
- pandas
- matplotlib
- scikit-learn (for metrics only)

---

## ✍️ Author
KareemShaik.com

---


