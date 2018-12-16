from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# qiita 画像とデータ
digits = load_digits()

plt.imshow(digits.images[0])
plt.axis('off')
plt.gray()
plt.show()

print(digits.images[0])

print(digits.data[0])

print(digits.target[0])