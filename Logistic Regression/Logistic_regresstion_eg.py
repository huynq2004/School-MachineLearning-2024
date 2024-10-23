# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2)


#Khởi tạo mảng dữ liệu đầu vào và đầu ra: 
#X là thuộc tính đầu vào (giờ học), Y là nhãn đầu ra (đỗ hay trượt)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extended data 
#Thêm 1 hàng toàn 1 vào ma trận thuộc tính đầu vào X
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

#Hàm kích hoạt 
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    #Trọng số khởi tạo cho thuộc tính
    it = 0          
    N = X.shape[1]  #Là số cột, tức là số bản ghi dữ liệu
    d = X.shape[0]
    count = 0       #đếm số lần lặp
    check_w_after = 20      #Kiểm tra lại tiêu chí dừng sau mỗi 20 lần cập nhật w, để tăng hiệu suất -> giảm vc tính toán mỗi lần 
    while count < max_count:
        # mix data -> trộn dữ liệu lẫn lộn các chỉ số để việc dự đoán k phụ thuộc vào thứ tự các bản ghi
        #Chọn mẫu theo thứ tự ngẫu nhiên mỗi lần huấn luyện
        mix_id = np.random.permutation(N)
        for i in mix_id:
            #định dạng lại 1 bản ghi thành vector cột có kích thước (d, 1) để nhân ma trận
            xi = X[:, i].reshape(d, 1)
            yi = y[i]   #nhãn của bản ghi thứ i
            zi = sigmoid(np.dot(w[-1].T, xi))   #Tính hàm sigmoid của bản ghi i -> Chuyển tổng trọng số thuộc tính đã tính thành giá trị khoảng [0;1]
            w_new = w[-1] + eta*(yi - zi)*xi    #Cập nhật trọng số mới = cũ + hàm mất mát
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                #độ chênh lệch giữa trọng số hiện tại w_new và trọng số cách đây 20 lần cập nhật
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

#In thử d ra -> Thấy d = 2, tức là X có 2 hàng (1 hàng bias đc thêm vào và 1 hàng giờ học là thuộc tính ban đầu )
#print(d)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])

#Biểu diễn kết quả trên đồ thị
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()

#predit func
def predict(input):
    input = np.concatenate((np.ones((1, input.shape[1])), input), axis = 0)
    z = sigmoid(np.dot(w[-1].T, input))
    outcome = []
    for i in range(z.shape[1]):
        if z[0, i]<0.5: outcome.append("trượt")
        else: outcome.append("đỗ")
    return outcome

input = np.array([[3.35, 1.62, 1.00, 4.65]])
print(predict(input))




