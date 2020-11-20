import sys
sys.path.append('../')
from sklearn.cluster import KMeans
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math 
import random as random
from MNIST.MNISTDataLoading import * 


class NearestNeighborClassifierMNISK:
    def __init__(self):
        pass

    def train_nearest_neighbor_k_neighbor(self, traning_set, k_neighbor):
        traning_images_bytes = [traning_set[i].raw_bytes for i in range(len(traning_set))]
        traning_images_labels = [traning_set[i].label for i in range(len(traning_set))]

        pca_tranning = PCA(n_components=2)
        pca_traning_values = pca_tranning.fit_transform(traning_images_bytes)

        k_neighbor_classifier = neighbors.KNeighborsClassifier(k_neighbor, weights='uniform')
        k_neighbor_classifier.fit(pca_traning_values, traning_images_labels)

        return k_neighbor_classifier


    def predict_test_data_k_neighbor(self, trained_model, testing_set):
        testing_images_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]
        #testing_images_labels = [traning_set[i].label for i in range(len(traning_set))]

        pca_testing = PCA(n_components=2)
        pca_testing_values = pca_testing.fit_transform(testing_images_bytes)

        predicted_values = trained_model.predict(pca_testing_values)
        return predicted_values


    def evaluate_succes_rate_k_neighbor(self, predict_labels, testing_set):
        testing_labels = [testing_set[i].label for i in range(len(testing_set))]
        print("Predict labels len : ", len(predict_labels) , " : " , predict_labels)
        print("Testing labels : ", testing_labels)
    
        correct_predicted = 0
        for i in range(len(predict_labels)):
            if predict_labels[i] == testing_labels[i]:
                correct_predicted +=1
    
        return correct_predicted


    def train_nearest_neighbor_model_k_means(self, traning_images):
        traning_images_bytes = [traning_images[i].raw_bytes for i in range(len(traning_images))]
        pca_tranning = PCA(n_components=2)
        pca_traning_values = pca_tranning.fit_transform(traning_images_bytes)

        max_k_cluster_size = len(pca_traning_values)

        # run KMeans so that each point gets its own centroid.
        kmeans = KMeans(n_clusters=max_k_cluster_size, random_state=0)
        kmeans.fit(pca_traning_values)

        return (kmeans, pca_traning_values)

    def predict_one_neighbor_class_k_means(self, kmeans_object, testing_set):
        testing_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]
        pca_images = PCA(n_components=2)
        pca_testing_image = pca_images.fit_transform(testing_bytes)
        predicted_labels = kmeans_object.predict(pca_testing_image)

        return predicted_labels

    def calculate_succes_rate_one_k_means(self, predicted_labels, traning_set, testing_set):
        succes_rate = 0
        for i, predicted_label in enumerate(predicted_labels):
            guessed_image = traning_set[predicted_label]
            if(testing_set[i].label == guessed_image.label):
                succes_rate +=1
        return succes_rate
    
    def plot_neighbor_classes_kmeans(self, kmeans_object, pca_values_image, predicted_data = []):
        plt.figure(figsize=(9, 9))
    
        cluster_size = len(pca_values_image)
  
        pca_X_vals_class = [pca_values_image[i][0] for i in range(len(pca_values_image))]
        pca_Y_vals_class = [pca_values_image[i][1] for i in range(len(pca_values_image))]
  
        center_X_vals = [kmeans_object.cluster_centers_[i][0] for i in range(len(kmeans_object.cluster_centers_))]
        center_Y_vals = [kmeans_object.cluster_centers_[i][1] for i in range(len(kmeans_object.cluster_centers_))]
  
        plt.scatter(center_X_vals, center_Y_vals, s=200, c='blue', label="centroids for each sample", alpha=0.2 )
        plt.scatter(pca_X_vals_class , pca_Y_vals_class, s=50, c="red", marker="*", label="each sample", zorder=10 )

        if  len(predicted_data) != 0:
            predicted_data_X_values = [predicted_data[i][0] for i in range(len(predicted_data))]
            predicted_data_Y_values = [predicted_data[i][1] for i in range(len(predicted_data))]
            plt.scatter(predicted_data_X_values , predicted_data_Y_values, s=200, marker="p", c="black",  label=f"predicted_test_data", zorder=10, alpha=0.7 )
            plt.title(f"K-means, k={cluster_size} - sub classes for class : {1} - predicted data")
        else:
            plt.title(f"K-means, k_max={cluster_size} - each sample has its own centroid")

        l = plt.legend()
        l.set_zorder(50)  # put the legend on top



if __name__ == '__main__': 
    # Fetch traning/testing images
    data_loader = MNISTDataLoader()
    testing_set = data_loader.fetch_testing_set()
    traning_set = data_loader.fetch_traning_set()
    # Create class for traning/testing
    nearest_neighbor_classifier_MNIST = NearestNeighborClassifierMNISK()

    trained_model  = nearest_neighbor_classifier_MNIST.train_nearest_neighbor_k_neighbor(traning_set, 2)
    predict_labels = nearest_neighbor_classifier_MNIST.predict_test_data_k_neighbor(trained_model, testing_set)
    succes_rate_k_2 = nearest_neighbor_classifier_MNIST.evaluate_succes_rate_k_neighbor(predict_labels, testing_set)

    print(f"succes rate k=2 : {succes_rate_k_2}/{len(testing_set)} = {succes_rate_k_2/len(testing_set)} = {succes_rate_k_2/len(testing_set) *100}")

    trained_model  = nearest_neighbor_classifier_MNIST.train_nearest_neighbor_k_neighbor(traning_set, 5)
    predict_labels = nearest_neighbor_classifier_MNIST.predict_test_data_k_neighbor(trained_model, testing_set)
    succes_rate_k_5 = nearest_neighbor_classifier_MNIST.evaluate_succes_rate_k_neighbor(predict_labels, testing_set)

    print(f"succes rate k=5 : {succes_rate_k_5}/{len(testing_set)} = {succes_rate_k_5/len(testing_set)} = {succes_rate_k_5/len(testing_set) *100}")

    trained_model  = nearest_neighbor_classifier_MNIST.train_nearest_neighbor_k_neighbor(traning_set, 10)
    predict_labels = nearest_neighbor_classifier_MNIST.predict_test_data_k_neighbor(trained_model, testing_set)
    succes_rate_k_10 = nearest_neighbor_classifier_MNIST.evaluate_succes_rate_k_neighbor(predict_labels, testing_set)

    print(f"succes rate k=10 : {succes_rate_k_10}/{len(testing_set)} = {succes_rate_k_10/len(testing_set)} = {succes_rate_k_10/len(testing_set) *100}")


    trained_model  = nearest_neighbor_classifier_MNIST.train_nearest_neighbor_k_neighbor(traning_set, 50)
    predict_labels = nearest_neighbor_classifier_MNIST.predict_test_data_k_neighbor(trained_model, testing_set)
    succes_rate_k_50 = nearest_neighbor_classifier_MNIST.evaluate_succes_rate_k_neighbor(predict_labels, testing_set)

    print(f"succes rate k=50 : {succes_rate_k_50}/{len(testing_set)} = {succes_rate_k_50/len(testing_set)}= {succes_rate_k_50/len(testing_set) *100}")
