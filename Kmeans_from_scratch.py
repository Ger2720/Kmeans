import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

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
        l_labels = []  # Initialise une liste pour stocker les étiquettes des clusters à chaque itération.
        iter_count = 0  # Compteur pour suivre le nombre d'itérations.

        # Initialize centroids
        self._initialize_centroids()  # Initialise les centroïdes en utilisant la méthode spécifiée (kmeans, kmeans++, soft_kmeans).

        while True:  # Boucle jusqu'à convergence.
            # Assign labels based on closest centroid
            self.labels = self._compute_labels()  # Attribue des étiquettes aux points de données en fonction du centroïde le plus proche.
            l_labels.append(self.labels)  # Ajoute l'ensemble actuel des étiquettes à la liste.

            # Check convergence
            if len(l_labels) > 2 and np.array_equal(l_labels[-1], l_labels[-2]):
                # Vérifie si les étiquettes n'ont pas changé par rapport à l'itération précédente, indiquant la convergence.
                self.centroids_etiq = {f"Centroid N°{i + 1}": self.centroids[i] for i in
                                       range(self.centroids.shape[
                                                 0])}  # Stocke les informations des centroïdes pour le reporting.
                break  # Sort de la boucle si le modèle a convergé.
            else:
                # Update centroids
                self.centroids = self._compute_centroids()  # Met à jour la position des centroïdes en fonction des étiquettes actuelles.
                iter_count += 1  # Incrémente le compteur d'itérations.

        self.labels = l_labels[-1]  # Stocke les étiquettes finales des clusters.
        print(f"Model {self.init} converged in {iter_count} iterations")  # Affiche le nombre d'itérations nécessaires à la convergence.

    def _initialize_centroids(self):
        random_state = np.random.RandomState(
            self.random_state)  # Crée un générateur de nombres aléatoires avec la graine spécifiée.

        if self.init == 'kmeans':
            i = random_state.permutation(self.data.shape[0])[
                :self.k]  # Sélectionne aléatoirement 'k' indices pour initialiser les centroïdes.
            self.centroids = self.data[i]  # Initialise les centroïdes avec ces points aléatoires.

        elif self.init in ['kmeans++', 'soft_kmeans']:
            i = random_state.permutation(self.data.shape[0])[
                0]  # Sélectionne un indice aléatoire pour le premier centroïde.
            self.centroids = self.data[i].reshape(1, -1)  # Initialise le premier centroïde.
            if self.init == 'soft_kmeans':
                self.var = np.var(self.data)  # Calcule la variance des données, utilisée dans Soft K-means.
            for _ in range(self.k - 1):
                # Calcule les distances entre chaque point de données et les centroïdes existants.
                distance = np.sqrt(((self.data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
                distance_min = np.min(distance, axis=0)  # Trouve la distance minimale pour chaque point.
                prob_to_choose = distance_min / np.sum(
                    distance_min)  # Calcule les probabilités de sélection pour le prochain centroïde.
                next_i = np.random.choice(self.data.shape[0], 1,
                                          p=prob_to_choose)  # Sélectionne le prochain centroïde basé sur ces probabilités.
                self.centroids = np.vstack(
                    (self.centroids, self.data[next_i].reshape(1, -1)))  # Ajoute le nouveau centroïde à la liste.

        print("Centroids initialization done!")  # Indique que l'initialisation des centroïdes est terminée.

    def _compute_labels(self):
        if self.init == 'soft_kmeans':
            labels, self.l_p = self._p_mat()  # Pour Soft K-means, utilise une matrice de probabilités pour déterminer les étiquettes.
        else:
            # Calcule la distance entre chaque point de données et tous les centroïdes.
            distance = np.sqrt(((self.data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distance, axis=0)  # Attribue chaque point au centroïde le plus proche.

        return labels  # Retourne les étiquettes calculées.

    def _compute_centroids(self):
        centroids = np.zeros(
            (self.k, self.data.shape[1]))  # Initialise un tableau pour stocker les nouveaux centroïdes.

        if self.init == 'soft_kmeans':
            # Pour Soft K-means, met à jour les centroïdes en tenant compte des probabilités.
            for j in range(self.k):
                sum_num = 0
                sum_denom = 0
                for i in np.where(self.labels == j)[0]:
                    sum_num += self.l_p[i] * self.data[i]  # Poids par la probabilité.
                    sum_denom += self.l_p[i]
                centroids[j, :] = sum_num / sum_denom  # Calcule le nouveau centroïde comme une moyenne pondérée.
        else:
            # Pour K-means et K-means++, calcule simplement la moyenne des points de chaque cluster.
            for i in range(self.k):
                centroids[i, :] = np.mean(self.data[self.labels == i, :], axis=0)

        return centroids  # Retourne les nouveaux centroïdes.

    def _p_mat(self):
        p_mat = np.zeros(
            (self.data.shape[0], self.k))  # Initialise une matrice de probabilités pour chaque point et chaque cluster.
        for i in range(p_mat.shape[0]):
            for k in range(self.k):
                # Calcule la probabilité d'appartenance de chaque point au cluster k en utilisant une fonction gaussienne.
                p_mat[i, k] = np.exp((-0.5 * ((self.data[i] - self.centroids[k]) ** 2).sum()) / self.var)
        # Retourne les étiquettes et les probabilités maximales pour chaque point.
        return np.argmax(p_mat, axis=1), np.max(p_mat, axis=1)

    def predict(self):
        self.predicted_labels = self._compute_labels()  # Calcule les étiquettes des clusters pour les points de données en utilisant la méthode actuelle de clustering.
        return self.predicted_labels  # Retourne les étiquettes prédites.

    def compute_silhouette(self):
        silhouette = []  # Initialise une liste pour stocker les scores de silhouette pour chaque point.
        for i in range(self.data.shape[0]):
            # Calcule la distance moyenne entre le point i et les autres points de son cluster.
            cluster_indices = self.labels == self.labels[i]
            cluster_points = self.data[cluster_indices]
            X_i_reshaped = self.data[i].reshape(1, -1)
            distances_intra = np.sqrt(((cluster_points - X_i_reshaped) ** 2).sum(axis=1))
            a_i = distances_intra.mean()

            distances_inter = []  # Liste pour stocker la distance moyenne aux points des autres clusters.
            for k in range(self.k):
                if k != self.labels[i]:
                    # Calcule la distance moyenne entre le point i et les points du cluster k.
                    cluster_points = self.data[self.labels == k]
                    distances_inter.append(np.mean(np.sqrt(((cluster_points - X_i_reshaped) ** 2).sum(axis=1))))

            b_i = min(distances_inter)  # Trouve la distance minimale entre i et les points d'un autre cluster.
            silhouette_i = (b_i - a_i) / max(a_i, b_i)  # Calcule le score de silhouette pour le point i.
            silhouette.append(silhouette_i)  # Ajoute le score de silhouette à la liste.

        self.silhouette = np.mean(silhouette, axis=0)  # Calcule le score de silhouette moyen pour tous les points.
        return self.silhouette  # Retourne le score de silhouette moyen.

    def visualize_data(self):
        palette = sns.color_palette("husl", self.k)  # Crée une palette de couleurs pour les différents clusters.

        plt.figure(figsize=(8, 6))  # Initialise une nouvelle figure pour le graphique avec une taille définie.
        if self.data.shape[1] == 2:  # Vérifie si les données sont en 2 dimensions (pour permettre la visualisation).
            for k in range(self.k):  # Boucle sur chaque cluster.
                # Trace les points de données de chaque cluster en utilisant des couleurs différentes.
                plt.scatter(self.data[self.labels == k, 0], self.data[self.labels == k, 1], color=palette[k],
                            label=f"Cluster {k + 1}")

            # Ajoute le titre, les étiquettes des axes et la légende au graphique.
            plt.title(f"{self.k} Clusters/Modèle {self.init}")
            plt.xlabel('Axe X')
            plt.ylabel('Axe Y')
            plt.legend()
            plt.show()  # Affiche le graphique.


#X, y = make_blobs(n_samples=10000, centers=10, cluster_std=20, random_state=0)


def testing(data, data_name, min_k, max_k):
    # Initialisation des listes pour stocker les scores de silhouette pour chaque modèle.
    scores_silhouette_kmeans = []
    scores_silhouette_kmeansplusplus = []
    scores_silhouette_softkmeans = []

    # Boucle sur une plage de valeurs de 'k' (nombre de clusters).
    for k in range(min_k, max_k):
        print(f"Essai avec {k} clusters")  # Affiche le nombre de clusters actuellement testé.

        # Création des instances des différents modèles de KMeans avec le nombre de clusters spécifié.
        KMeans_inst = KMeans(data=data, k=k)
        KMeansplusplus_inst = KMeans(data=data, k=k, init='kmeans++')
        soft_kmeans_inst = KMeans(data=data, k=k, init='soft_kmeans')

        # Exécution de la méthode fit pour chaque modèle.
        KMeans_inst.fit()
        KMeansplusplus_inst.fit()
        soft_kmeans_inst.fit()

        # Calcul et affichage du score de silhouette pour chaque modèle.
        KMeans_inst.compute_silhouette()
        print(f"Score Silhouette du modèle KMeans basique: {KMeans_inst.silhouette}")
        KMeansplusplus_inst.compute_silhouette()
        print(f"Score Silhouette du modèle KMeans++: {KMeansplusplus_inst.silhouette}")
        soft_kmeans_inst.compute_silhouette()
        print(f"Score Silhouette du modèle Soft KMeans: {soft_kmeans_inst.silhouette}")

        # Ajout des scores de silhouette aux listes correspondantes.
        scores_silhouette_kmeans.append(KMeans_inst.silhouette)
        scores_silhouette_kmeansplusplus.append(KMeansplusplus_inst.silhouette)
        scores_silhouette_softkmeans.append(soft_kmeans_inst.silhouette)

    # Préparation des données pour la visualisation.
    clusters = range(min_k, max_k)

    # Configuration et affichage du graphique.
    plt.figure(figsize=(10, 6))
    plt.plot(clusters, scores_silhouette_kmeans, label='KMeans')
    plt.plot(clusters, scores_silhouette_kmeansplusplus, label='KMeans++')
    plt.plot(clusters, scores_silhouette_softkmeans, label='Soft KMeans')
    plt.title(f"Evolution du Score Silhouette en fonction du Nombre de Clusters pour le dataset {data_name}")
    plt.xlabel('Nombre de Clusters')
    plt.ylabel('Score Silhouette')
    plt.legend()
    plt.show()  # Affiche le graphique comparant les scores de silhouette.
