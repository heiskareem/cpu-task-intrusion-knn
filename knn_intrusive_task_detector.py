import pandas as pd
from collections import Counter
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


pd.read_csv("cpu_tasks.csv")


# KNN functions remain the same
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def knn_classify(test_point, train_data, k=5):
    distances = []
    test_features = test_point[:3]
    for _, row in train_data.iterrows():
        train_features = row[['CPU_Usage', 'Priority', 'Order']].tolist()
        distance = euclidean_distance(test_features, train_features)
        distances.append((distance, row['Label']))
    distances.sort(key=lambda x: x[0])
    top_k = [label for _, label in distances[:k]]
    most_common = Counter(top_k).most_common(1)[0][0]
    return most_common

#  Generate dataset and split
dataset_corr = pd.read_csv("cpu_tasks.csv")
train_df_corr = dataset_corr.sample(frac=0.8, random_state=42)
test_df_corr = dataset_corr.drop(train_df_corr.index)

#  Run KNN
predictions_corr = []
for _, row in test_df_corr.iterrows():
    test_point = row[['CPU_Usage', 'Priority', 'Order']].tolist()
    predicted_label = knn_classify(test_point, train_df_corr, k=5)
    predictions_corr.append(predicted_label)
    
    
    
test_df_corr

predictions_corr


# Evaluate and show confusion matrix
test_df_corr['Predicted'] = predictions_corr
cm = confusion_matrix(test_df_corr['Label'], test_df_corr['Predicted'], labels=["intrusive", "non-intrusive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Intrusive", "Non-Intrusive"])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - KNN Intrusive Task Detection")
plt.grid(False)
plt.show()


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_df_corr['Label'], test_df_corr['Predicted'])

accuracy