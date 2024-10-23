import numpy as np 
import imageio  # Thay thế scipy.misc bằng imageio để đọc ảnh
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = 'Principal Component Analysis/unpadded/'  # Đường dẫn tới cơ sở dữ liệu 
ids = range(1, 16)  # 15 người
states = ['centerlight', 'glasses', 'happy', 'leftlight', 
          'noglasses', 'normal', 'rightlight', 'sad', 
          'sleepy', 'surprised', 'wink' ]
prefix = 'subject'
surfix = '.pgm'

h = 116  # Chiều cao ảnh
w = 98   # Chiều rộng ảnh
D = h * w
N = len(states) * 15  # Tổng số ảnh

X = np.zeros((D, N))

# Thu thập toàn bộ dữ liệu
cnt = 0 
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        X[:, cnt] = imageio.imread(fn).reshape(D)  # Dùng imageio.imread thay thế
        cnt += 1 

# Thực hiện PCA, chú ý rằng mỗi hàng là một điểm dữ liệu
pca = PCA(n_components=100)  # k = 100
pca.fit(X.T)

# Ma trận chiếu
U = pca.components_.T

# Hiển thị các eigenface
for i in range(U.shape[1]):
    plt.axis('off')
    f1 = plt.imshow(U[:, i].reshape(116, 98), interpolation='nearest')
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
    plt.gray()
    fn = 'Principal Component Analysis/eigenfaces/eigenface' + str(i).zfill(2) + '.png'
    plt.savefig(fn, bbox_inches='tight', pad_inches=0)

# Xem tái tạo ảnh của 6 người đầu tiên
for person_id in range(1, 7):
    for state in ['centerlight']:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)  # Dùng imageio.imread thay thế
        plt.axis('off')
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'Principal Component Analysis/ori/ori' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()
        
        # Chuẩn hóa và trừ đi trung bình, không quên
        x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
        # Mã hóa
        z = U.T.dot(x)
        # Giải mã
        x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

        # Tái tạo lại kích thước ảnh ban đầu
        im_tilde = x_tilde.reshape(116, 98)
        plt.axis('off')
        f1 = plt.imshow(im_tilde, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'Principal Component Analysis/res/res' + str(person_id).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()

# Hiển thị toàn bộ các trạng thái của một người (ví dụ: người thứ 10)
cnt = 0 
for person_id in [10]:
    for ii, state in enumerate(states):
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)  # Dùng imageio.imread thay thế
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)

        fn = 'Principal Component Analysis/ex/ex' + str(ii).zfill(2) + '.png'
        plt.axis('off')
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()
