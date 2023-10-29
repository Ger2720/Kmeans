import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, random_state=42, init='kmeans', debug=False):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.centroids_etiq = None
        self.predicted_labels = None
        self.init = init
        self.debug = debug

    def fit(self, X):
        l_labels = []
        iter_count = 0

        # Initialize centroids
        self._initialize_centroids(X)

        while True:
            # Assign labels based on closest centroid
            labels = self._compute_labels(X)
            l_labels.append(labels)

            # Check convergence
            if len(l_labels) > 2 and np.array_equal(l_labels[-1], l_labels[-2]):
                self.centroids_etiq = {f"Centroid N°{i + 1}": self.centroids[i] for i in
                                       range(self.centroids.shape[0])}
                break
            else:
                # Update centroids
                self.centroids = self._compute_centroids(X, labels)
                iter_count += 1

        print(f"Model {self.init} converged in {iter_count} iterations")

    def _initialize_centroids(self, X):
        random_state = np.random.RandomState(self.random_state)

        if self.init == 'kmeans':
            i = random_state.permutation(X.shape[0])[:self.k]
            self.centroids = X[i]
        elif self.init == 'kmeans++':
            i = random_state.permutation(X.shape[0])[0]
            self.centroids = X[i].reshape(1, -1)
            for _ in range(self.k - 1):
                distance = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
                distance_min = np.min(distance, axis=0)
                prob_to_choose = distance_min / np.sum(distance_min)
                next_i = np.random.choice(X.shape[0], 1, p=prob_to_choose)
                self.centroids = np.vstack((self.centroids, X[next_i].reshape(1, -1)))

            print("All Centroids initialized")

    def _compute_labels(self, X):
        distance = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distance, axis=0)

    def _compute_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            centroids[i, :] = np.mean(X[labels == i, :], axis=0)
        return centroids

    def predict(self, X):
        self.predicted_labels = self._compute_labels(X)
        return self.predicted_labels


X_train = np.random.randint(0, 100, 1000000).reshape(-1, 10)
KMeans_inst = KMeans(k= 10)
KMeansplusplus_inst = KMeans(k= 10, init= 'kmeans++')
KMeansplusplus_inst.fit(X_train)
KMeans_inst.fit(X_train)

X_train = np.random.randint(0, 100, 100000).reshape(-1, 10)
X_test = np.random.randint(0, 100, 100).reshape(-1, 10)
KMeans_inst.fit(X_train)
KMeansplusplus_inst.fit(X_train)
KMeans_inst.predict(X_test)
KMeans_inst.centroids.shape
KMeans_inst.centroids_etiq
KMeans_inst.predicted_labels




