import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, data, k=3, random_state=42, init='kmeans', debug=False):
        self.k = k
        self.data = data
        self.random_state = random_state
        self.centroids = None
        self.centroids_etiq = None
        self.predicted_labels = None
        self.init = init
        self.debug = debug
        self.labels = None
        self.silhouette = None
        self.var = None
        self.l_p = None

    def fit(self):
        l_labels = []
        iter_count = 0

        # Initialize centroids
        self._initialize_centroids()

        while True:
            # Assign labels based on closest centroid
            self.labels = self._compute_labels()
            l_labels.append(self.labels)

            # Check convergence
            if len(l_labels) > 2 and np.array_equal(l_labels[-1], l_labels[-2]):
                self.centroids_etiq = {f"Centroid N°{i + 1}": self.centroids[i] for i in
                                       range(self.centroids.shape[0])}
                break
            else:
                # Update centroids
                self.centroids = self._compute_centroids()
                iter_count += 1
        self.labels = l_labels[-1]
        print(f"Model {self.init} converged in {iter_count} iterations")

    def _initialize_centroids(self):
        random_state = np.random.RandomState(self.random_state)

        if self.init == 'kmeans':
            i = random_state.permutation(self.data.shape[0])[:self.k]
            self.centroids = self.data[i]

        elif self.init in ['kmeans++', 'soft_kmeans']:
            i = random_state.permutation(self.data.shape[0])[0]
            self.centroids = self.data[i].reshape(1, -1)
            if self.init == 'soft_kmeans':
                self.var = np.var(self.data)
            for _ in range(self.k - 1):
                distance = np.sqrt(((self.data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
                distance_min = np.min(distance, axis=0)
                prob_to_choose = distance_min / np.sum(distance_min)
                next_i = np.random.choice(self.data.shape[0], 1, p=prob_to_choose)
                self.centroids = np.vstack((self.centroids, self.data[next_i].reshape(1, -1)))

            print("Centroids initialization done!")

    def _compute_labels(self):
        if self.init == 'soft_kmeans':
            labels, self.l_p =  self._p_mat()
        else:
            distance = np.sqrt(((self.data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distance, axis=0)

        return labels
    def _compute_centroids(self):
        centroids = np.zeros((self.k, self.data.shape[1]))
        if self.init == 'soft_kmeans':
            for j in range(self.k):
                sum_num = 0
                sum_denom = 0
                for i in np.where(self.labels == j)[0]:
                    sum_num += self.l_p[i]*self.data[i]
                    sum_denom += self.l_p[i]
                centroids[j, :] = sum_num/sum_denom
        else:
            for i in range(self.k):
                centroids[i, :] = np.mean(self.data[self.labels == i, :], axis=0)
        return centroids

    def _p_mat(self):
        p_mat = np.zeros((self.data.shape[0], self.k))
        for i in range(p_mat.shape[0]):
            for k in range(self.k):
                p_mat[i, k]= np.exp((-0.5*((self.data[i] - self.centroids[k])**2).sum())/self.var)
        return np.argmax(p_mat, axis=1), np.max(p_mat, axis=1)

    def predict(self):
        self.predicted_labels = self._compute_labels()
        return self.predicted_labels

    def compute_silhouette(self):
        silhouette = []
        for i in range(self.data.shape[0]):
            cluster_indices = self.labels == self.labels[i]
            cluster_points = self.data[cluster_indices]
            X_i_reshaped = self.data[i].reshape(1, -1)
            distances_intra = np.sqrt(((cluster_points - X_i_reshaped) ** 2).sum(axis=1))
            a_i = distances_intra.mean()
            distances_inter = []
            for k in range(self.k):
                if k != self.labels[i]:
                    cluster_points = self.data[self.labels == k]
                    distances_inter.append(np.mean(np.sqrt(((cluster_points - X_i_reshaped) ** 2).sum(axis=1))))

            b_i = min(distances_inter)
            silhouette_i = (b_i - a_i)/max(a_i, b_i)
            silhouette.append(silhouette_i)
            #if len(silhouette) % 100 ==0:
               # print("\033[H\033[J", end="")

                #print(f"nombre de points traités :{len(silhouette)}, il reste {X.shape[0]-len(silhouette)} à traiter")


        self.silhouette = np.mean(silhouette, axis=0)
        return self.silhouette



X, y = make_blobs(n_samples=10000, centers=10, cluster_std=20, random_state=0)


for k in range(2, 25):
    print(f"Essai avec {k} clusters")
    KMeans_inst = KMeans(data=X, k=k)
    KMeansplusplus_inst = KMeans(data=X, k=k, init='kmeans++')
    soft_kmeans_inst = KMeans(data=X, k=k, init= 'soft_kmeans')
    KMeans_inst.fit()
    KMeansplusplus_inst.fit()
    soft_kmeans_inst.fit()
    KMeans_inst.compute_silhouette()
    print(f"Score Silhouette du modèle KMeans basique: {KMeans_inst.silhouette}")
    KMeansplusplus_inst.compute_silhouette()
    print(f"Score Silhouette du modèle KMeans++: {KMeansplusplus_inst.silhouette}")
    soft_kmeans_inst.compute_silhouette()
    print(f"Score Silhouette du modèle Soft KMeans: {soft_kmeans_inst.silhouette}")




