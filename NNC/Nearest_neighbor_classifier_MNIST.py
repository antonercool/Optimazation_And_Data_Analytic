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

    # Train the traing_set with k number of neighbors
    def train_nearest_neighbor_k_neighbor(self, traning_set, k_neighbor):
        traning_images_bytes = [traning_set[i].raw_bytes for i in range(len(traning_set))]
        traning_images_labels = [traning_set[i].label for i in range(len(traning_set))]

        pca_tranning = PCA(n_components=2)
        pca_traning_values = pca_tranning.fit_transform(traning_images_bytes)

        k_neighbor_classifier = neighbors.KNeighborsClassifier(k_neighbor, weights='uniform')
        k_neighbor_classifier.fit(pca_traning_values, traning_images_labels)

        return k_neighbor_classifier

    # Predics the test data, using the trained model
    def predict_test_data_k_neighbor(self, trained_model, testing_set):
        testing_images_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]

        pca_testing = PCA(n_components=2)
        pca_testing_values = pca_testing.fit_transform(testing_images_bytes)

        predicted_values = trained_model.predict(pca_testing_values)
        return predicted_values

    # Evalulates the precision rate of the trained model
    def evaluate_succes_rate_k_neighbor(self, predict_labels, testing_set):
        testing_labels = [testing_set[i].label for i in range(len(testing_set))]
    
        correct_predicted = 0
        for i in range(len(predict_labels)):
            if predict_labels[i] == testing_labels[i]:
                correct_predicted +=1
    
        return correct_predicted



if __name__ == '__main__': 
    # Fetch traning/testing images
    data_loader = MNISTDataLoader()
    testing_set = data_loader.fetch_testing_set()
    traning_set = data_loader.fetch_traning_set()
    # Create class for traning/testing
    nearest_neighbor_classifier_MNIST = NearestNeighborClassifierMNISK()

    
    trained_model  = nearest_neighbor_classifier_MNIST.train_nearest_neighbor_k_neighbor(traning_set, 1)
    predict_labels = nearest_neighbor_classifier_MNIST.predict_test_data_k_neighbor(trained_model, testing_set)
    succes_rate_k_1 = nearest_neighbor_classifier_MNIST.evaluate_succes_rate_k_neighbor(predict_labels, testing_set)

    print(f"succes rate k=1 : {succes_rate_k_1}/{len(testing_set)} = {succes_rate_k_1/len(testing_set)} = {succes_rate_k_1/len(testing_set) *100}")

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
