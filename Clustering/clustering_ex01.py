import numpy as np
import matplotlib.pyplot as plt

# Các điểm dữ liệu
X = np.array([(1, 4), (1, 6), (2, 6), (3, 8), (4, 3), (5, 2)])

# Số cụm cần phân
K = 2

# Khởi tạo các tâm cụm ban đầu là x1 và x3
initial_centers = np.array([X[0], X[2]])  # Chọn điểm (1, 4) và (2, 6) làm tâm khởi đầu

def kmeans_assign_labels(X, centers):
    # Tính khoảng cách từ mỗi điểm đến các tâm
    D = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Khoảng cách từ từng điểm đến các tâm
    return np.argmin(D, axis=1)  # Gán mỗi điểm vào cụm có tâm gần nhất

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # Lấy các điểm thuộc cụm k
        Xk = X[labels == k]
        # Tính tâm mới của cụm k
        if len(Xk) > 0:  # Kiểm tra nếu có điểm trong cụm
            centers[k] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    # Kiểm tra xem các tâm có thay đổi hay không
    return np.array_equal(centers, new_centers)

def kmeans(X, K, initial_centers):
    centers = initial_centers
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers, new_centers):
            break
        centers = new_centers
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K, initial_centers)
print('Các tâm của 2 cụm:')
print(centers)
print(f'Số lần lặp: {it}')

# Vẽ biểu đồ
def kmeans_display(X, labels, centers):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=100, alpha=0.6, label='Điểm dữ liệu')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Tâm cụm')
    plt.title('Phân cụm K-Means')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

kmeans_display(X, labels[-1], centers)
