#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
using namespace std;

struct Task {
    int cpu_usage;
    int priority;
    int order;
    string label;
};

// Calculate Euclidean distance between two tasks
double euclidean_distance(const Task& a, const Task& b) {
    return sqrt(pow(a.cpu_usage - b.cpu_usage, 2) +
                pow(a.priority - b.priority, 2) +
                pow(a.order - b.order, 2));
}

// Perform KNN classification
string knn_classify(const Task& test_task, const vector<Task>& train_data, int k = 5) {
    vector<pair<double, string>> distances;
    for (const auto& train_task : train_data) {
        double dist = euclidean_distance(test_task, train_task);
        distances.push_back({dist, train_task.label});
    }
    sort(distances.begin(), distances.end());

    map<string, int> label_count;
    for (int i = 0; i < k; ++i) {
        label_count[distances[i].second]++;
    }

    string most_common;
    int max_count = 0;
    for (const auto& pair : label_count) {
        if (pair.second > max_count) {
            most_common = pair.first;
            max_count = pair.second;
        }
    }
    return most_common;
}

// Load dataset from CSV
vector<Task> load_csv(const string& filename) {
    vector<Task> data;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    while (getline(file, line)) {
        stringstream ss(line);
        string cpu, prio, ord, lbl;
        getline(ss, cpu, ',');
        getline(ss, prio, ',');
        getline(ss, ord, ',');
        getline(ss, lbl, ',');
        data.push_back({stoi(cpu), stoi(prio), stoi(ord), lbl});
    }
    return data;
}

int main() {
    vector<Task> dataset = load_csv("cpu_tasks.csv");
    int split_index = dataset.size() * 0.8;

    vector<Task> train_data(dataset.begin(), dataset.begin() + split_index);
    vector<Task> test_data(dataset.begin() + split_index, dataset.end());

    int correct = 0;
    map<string, map<string, int>> confusion_matrix;

    for (const auto& test_task : test_data) {
        string predicted = knn_classify(test_task, train_data);
        confusion_matrix[test_task.label][predicted]++;
        if (predicted == test_task.label) correct++;
    }

    // Print accuracy
    double accuracy = static_cast<double>(correct) / test_data.size();
    cout << "Accuracy: " << accuracy << endl;

    // Print confusion matrix
    cout << "\nConfusion Matrix:\n";
    cout << "\tPredicted: Intrusive\tPredicted: Non-Intrusive\n";
    for (const string& actual : {"intrusive", "non-intrusive"}) {
        cout << "Actual: " << actual << "\t" << confusion_matrix[actual]["intrusive"] << "\t\t\t" << confusion_matrix[actual]["non-intrusive"] << endl;
    }

    return 0;
}
