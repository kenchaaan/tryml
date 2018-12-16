from sklearn.datasets import load_digits
import numpy as np
import math


class MyKMeans:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers = []
        self.labels = []

    def _generate_random_num_according_to_given_pd(self, pd):
        cumulative_dist = np.cumsum(pd).tolist()
        cumulative_dist[-1] = 1.0
        random_num = np.random.rand()
        cumulative_dist.append(random_num)
        return sorted(cumulative_dist).index(random_num)

    def _make_pd_with_weight_of_power_of_distance(self, data, centers):
        list_of_distances = list(map(
            lambda z: z ** 2, list(map(
                lambda x: min(
                    list(map(lambda y: np.linalg.norm(y - x), centers))
                ), data
            ))))
        denominator = sum(list_of_distances)
        pd = list(map(
            lambda distance: distance / denominator, list_of_distances
        ))
        return pd

    def _choose_centers_at_random(self, data):
        self.labels = [math.floor(np.random.rand() * self.n_clusters) for i in range(len(data))]

    def _choose_centers_as_kmeansplus2(self, data):
        # see http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
        self.labels = [0.0 for i in range(len(data))]
        self.cluster_centers.append(data[math.floor(np.random.rand() * len(data))].tolist())
        while len(self.cluster_centers) < self.n_clusters:
            pd = self._make_pd_with_weight_of_power_of_distance(data, self.cluster_centers)
            self.cluster_centers.append(data[self._generate_random_num_according_to_given_pd(pd)].tolist())
        for k in range(len(data)):
            self.labels[k] = self.cluster_centers.index(
                min(self.cluster_centers, key=lambda j: np.linalg.norm(j - data[k]))
            )

    def fit(self, data, itr=1, algorithm='random'):
        if algorithm == 'random':
            self._choose_centers_at_random(data)
        elif algorithm == 'k-means++':
            self._choose_centers_as_kmeansplus2(data)
        for times in range(itr):
            cluster_centers_indices_ = []
            self.cluster_centers = []
            for j in range(self.n_clusters):
                cluster_centers_indices_.append([x for x, y in enumerate(self.labels) if y == j])
            for indices in cluster_centers_indices_:
                self.cluster_centers.append(
                    np.mean(np.array(list(map(lambda index: data[index], indices))), axis=0).tolist())
            for k in range(len(data)):
                self.labels[k] = self.cluster_centers.index(
                    min(self.cluster_centers, key=lambda x: np.linalg.norm(x - data[k])))


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.cluster import KMeans
    import collections

    # いくつの数字を識別するか
    nn = 5
    digits = load_digits(n_class=nn)

    # インスタンス初期化
    # mykm: 自作k-means(k-means++)
    # km: sklearnのk-means(k-means++)
    # mykm: 自作k-means(普通のk-means)
    mykm = MyKMeans(n_clusters=nn)
    mykm.fit(digits.data, itr=300, algorithm='k-means++')
    km = KMeans(n_clusters=nn)
    km.fit(digits.data)
    mykmr = MyKMeans(n_clusters=nn)
    mykmr.fit(digits.data, itr=300)

    # 結果。それぞれ180にちかければOK
    result_mykm = collections.Counter(mykm.labels)
    result_km = collections.Counter(km.labels_)
    result_mykmr = collections.Counter(mykmr.labels)
    print('result_mykm: ',result_mykm )
    print('result_km: ', result_km)
    print('result_mykmr: ', result_mykmr)

    # entropyを計算。math.log(nn)に近いほどOK
    def entropy(pd):
        return sum(list(map(lambda p: - p * math.log(p) if p != 0.0 else 0.0, pd)))
    pd_mykm = list(map(lambda x: x / len(digits.data), result_mykm.values()))
    pd_km = list(map(lambda x: x / len(digits.data), result_km.values()))
    pd_mykmr = list(map(lambda x: x / len(digits.data), result_mykmr.values()))
    print(
        entropy(pd_mykm),
        entropy(pd_km),
        entropy(pd_mykmr),
        math.log(nn)
    )

    # show
    n = 50
    num = 10
    gs = gridspec.GridSpec(mykm.n_clusters, num)
    plt.gray()
    x = {}
    for t in range(mykm.n_clusters):
        x[t] = 0
    for i in range(n):
        plt.subplot(gs[mykm.labels[i], x[mykm.labels[i]]])
        plt.imshow(digits.images[i])
        plt.axis('off')
        x[mykm.labels[i]] += 1
        if x[mykm.labels[i]] >= num:
            break
    plt.show()

    # セントロイド
    centroid_x = np.array(list(map(lambda x: round(x), mykm.cluster_centers[0])))
    centroid_y = np.array(list(map(lambda x: round(x), mykm.cluster_centers[1])))
    X = np.reshape(centroid_x, [8, 8])
    Y = np.reshape(centroid_y, [8, 8])
    print('X', X.tolist())
    print('Y', Y.tolist())
    plt.subplot(211)
    plt.imshow(X)
    plt.subplot(212)
    plt.imshow(Y)
    plt.show()
