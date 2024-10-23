import numpy as np
import pandas as pd

# Tính toán Gini cho phân loại
def gini_index(groups, classes):
    total_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / float(size)
            score += proportion * proportion
        gini += (1.0 - score) * (size / total_instances)
    return gini

# Chia tập dữ liệu dựa trên giá trị cột (feature)
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Chọn điểm chia (split) tốt nhất
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Xây dựng nút lá
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Tách nhánh và xây dựng cây
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Xây dựng cây quyết định
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Dự đoán kết quả với cây quyết định
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Đọc dữ liệu từ file weather.csv
df = pd.read_csv('Decision Tree\weather.csv')
# Giả sử cột cuối cùng là nhãn (target), và các cột còn lại là đặc trưng (features)
dataset = df.values.tolist()

# Chuyển đổi các giá trị không phải số thành số (nếu có)
from sklearn.preprocessing import LabelEncoder

label_encoders = []
for i in range(len(dataset[0]) - 1):
    le = LabelEncoder()
    feature_column = [row[i] for row in dataset]
    le.fit(feature_column)
    label_encoders.append(le)
    for row in dataset:
        row[i] = le.transform([row[i]])[0]

# Xây dựng cây quyết định
tree = build_tree(dataset, 3, 1)

# Dự đoán cho toàn bộ dữ liệu
predictions = [predict(tree, row) for row in dataset]
print('Dự đoán:', predictions)

# Dùng thư viện sklearn để kiểm tra
from sklearn.tree import DecisionTreeClassifier

# Chuẩn bị dữ liệu
X = [row[:-1] for row in dataset]
y = [row[-1] for row in dataset]

# Xây dựng cây quyết định bằng scikit-learn
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Dự đoán
prediction_sklearn = model.predict(X)
print('Dự đoán (sklearn):', prediction_sklearn)