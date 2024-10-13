import numpy as np

#Tính toán Gini cho phân loại
def gini_index(groups, classes):
    # Lấy tổng số điểm dữ liệu trong các nhóm
    total_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        # Tính xác suất cho mỗi lớp
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / float(size)
            score += proportion * proportion
        # Trọng số nhóm theo kích thước của nó
        gini += (1.0 - score) * (size / total_instances)
    return gini

#Chia tập dữ liệu dựa trên giá trị cột (feature)
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

#Chọn điểm chia (split) tốt nhất
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

#Xây dựng nút lá
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#Tách nhánh và xây dựng cây
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # Kiểm tra nếu không có nhánh nào
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Kiểm tra độ sâu tối đa
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Xử lý nhánh bên trái
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # Xử lý nhánh bên phải
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

#Xây dựng cây quyết định
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


#Dự đoán kết quả với cây quyết định
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


#example
# Dataset ví dụ
dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

# Xây dựng cây quyết định
tree = build_tree(dataset, 3, 1)

# Dự đoán cho một dòng dữ liệu
prediction = predict(tree, [6.642287351, 3.319983761])
print('Dự đoán:', prediction)

#Dùng thư viện
from sklearn.tree import DecisionTreeClassifier

# Chuẩn bị dữ liệu
X = [row[:-1] for row in dataset]
y = [row[-1] for row in dataset]

# Xây dựng cây quyết định bằng scikit-learn
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Dự đoán
prediction_sklearn = model.predict([[6.642287351, 3.319983761]])
print('Dự đoán (sklearn):', prediction_sklearn)
