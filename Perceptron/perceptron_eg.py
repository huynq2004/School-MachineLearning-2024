import numpy as np
import matplotlib.pyplot as plt

def predict(w, X):
    '''
    predict label of each row of X, given w
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    w: a 1-d numpy array of shape (d)
    '''
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    '''
    perform perceptron learning algorithm
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    y: a 1-d numpy array of shape (N), label of each row of X. y[i] = 1/-1
    w_init: a 1-d numpy array of shape (d)
    '''
    w = w_init
    iteration = 0
    while True:
        pred = predict(w, X)
        # find indexes of misclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        # number of misclassified points
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:  # no more misclassified points
            return w
        # random pick one misclassified point
        random_id = np.random.choice(mis_idxs, 1)[0]
        # update w
        w = w + y[random_id]*X[random_id]

        # In vector w sau mỗi vòng lặp
        iteration += 1
        print(f"Iteration {iteration}: w = {w}")
        
        # Vẽ minh họa sau mỗi vòng lặp
        plot_perceptron(X, y, w)

def plot_perceptron(X, y, w):
    # Vẽ các điểm dữ liệu
    X_pos = X[y == 1]
    X_neg = X[y == -1]
    
    plt.scatter(X_pos[:, 1], X_pos[:, 2], color='blue', label='Positive')
    plt.scatter(X_neg[:, 1], X_neg[:, 2], color='red', label='Negative')

    # Vẽ đường phân cách
    x_vals = np.linspace(-2, 2, 100)
    y_vals = -(w[1]/w[2]) * x_vals - (w[0]/w[2])  # Dạng đường phân cách
    plt.plot(x_vals, y_vals, label='Decision Boundary')
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()

# Tạo dữ liệu
means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))

# Thêm cột bias vào X
Xbar = np.concatenate((np.ones((2*N, 1)), X), axis=1)
w_init = np.random.randn(Xbar.shape[1])

# Chạy thuật toán Perceptron
w = perceptron(Xbar, y, w_init)
