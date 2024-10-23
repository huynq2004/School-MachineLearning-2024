import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu đầu vào
X = np.array([[60, 2, 10],   # Diện tích, Số phòng ngủ, Cách trung tâm
              [40, 2, 5],
              [100, 3, 7]])

# Vector giá y
y = np.array([[10],          # Giá tương ứng với các căn nhà
              [12], 
              [20]])

# Thêm cột bias (cột 1) vào X
one = np.ones((X.shape[0], 1))  # Cột toàn 1 để tính hệ số w_0
Xbar = np.concatenate((one, X), axis=1)

# Tính toán w bằng công thức w = (X.T * X)^(-1) * X.T * y
A = np.dot(Xbar.T, Xbar)        # X.T * X
b = np.dot(Xbar.T, y)           # X.T * y
w = np.dot(np.linalg.pinv(A), b)  # Tính w = A^(-1) * b

print('w = ', w)

# Dự đoán giá của căn nhà x = (50, 2, 8)
x_new = np.array([1, 50, 2, 8])  # Thêm 1 ở đầu cho bias
y_pred = np.dot(x_new, w)        # Tính giá dự đoán

print(f'Dự đoán giá của căn nhà với x = (50, 2, 8): {y_pred[0]:.2f}')

# Visualize dữ liệu
plt.plot(X[:, 0], y, 'ro')  # Dữ liệu thực tế (diện tích vs giá)
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá')
plt.title('Giá vs diện tích')
plt.show()

# Chuẩn bị dữ liệu để vẽ đường hồi quy cho diện tích
x0 = np.linspace(30, 110, 100)  # Dải diện tích từ 30 đến 110 m²
# Sử dụng số phòng ngủ = 2 và cách trung tâm = 7 để vẽ đường hồi quy
y0 = w[0] + w[1] * x0 + w[2] * 2 + w[3] * 7

# Vẽ dữ liệu và đường hồi quy
plt.plot(X[:, 0], y, 'ro')  # Dữ liệu thực tế
plt.plot(x0, y0, label="Fitting line")  # Đường hồi quy
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá')
plt.title('Diện tích với giá nhà')
plt.legend()
plt.show()

from sklearn import linear_model

# fit the model by Linear Regression using scikit-learn
regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept=False để tính toán bias
regr.fit(Xbar, y)

# Compare two results
print('Hệ số tìm bằng scikit-learn  : ', regr.coef_)
print('Hệ số tìm thủ công: ', w.T)