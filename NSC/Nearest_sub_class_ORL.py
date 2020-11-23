import sys
sys.path.append('../')
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random as random
from ORL.ORLLoading import * 

class NearestSubClassClassifier:
    def __init__(self):
        pass

    # Uses KMeans to distrubute the training images into sub classes for size k
    def find_sub_classes(self, images, k_cluster_size):
        images_bytes = [images[i].raw_bytes for i in range(len(images))]
        
        # Use PCA to transform each image into 2-d
        pca_images = PCA(n_components=2)
        pca_values_image = pca_images.fit_transform(images_bytes)

        kmeans = KMeans(n_clusters=k_cluster_size, random_state=0)
        kmeans.fit(pca_values_image)
        return (kmeans, pca_values_image, k_cluster_size)

    # Predict testing images of class k, into sub classes m
    def predict_sub_class(self, kmeans_object, testing_set):
        testing_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]
        pca_images = PCA(n_components=2)
        pca_testing_image = pca_images.fit_transform(testing_bytes)
        predicted_labels = kmeans_object.predict(pca_testing_image)

        print(f"Predicted data for label {testing_set[0].label} : {predicted_labels}");

        return pca_testing_image

    # Plot the sub classes from KMeans
    def plot_sub_classes_kmeans(self, kmeans_object, pca_values_image, label, cluster_size, predicted_data = []):
        plt.figure(figsize=(9, 9))
    
        pca_values_list = []

        for i in range(cluster_size): 
            temp_pca_value = []       
            for index, label_value in enumerate(kmeans_object.labels_):
                if label_value == i:
                    temp_pca_value.append(pca_values_image[index]) 
            pca_values_list.append(temp_pca_value)        

        for cluster_class_index in range(cluster_size):
            current_pca_X_vals_class = [pca_values_list[cluster_class_index][i][0] for i in range(len(pca_values_list[cluster_class_index]))]
            current_pca_Y_vals_class = [pca_values_list[cluster_class_index][i][1] for i in range(len(pca_values_list[cluster_class_index]))]
            plt.scatter(kmeans_object.cluster_centers_[cluster_class_index][0],kmeans_object.cluster_centers_[cluster_class_index][1], s=300, c='yellow', alpha=0.7)

            number_of_colors = 40
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
            plt.scatter(current_pca_X_vals_class , current_pca_Y_vals_class, s=150, c=color[cluster_class_index],  label=f"class-{cluster_class_index}", zorder=10, alpha=0.7 )
            plt.annotate(f"Centroid-class-{cluster_class_index}", (kmeans_object.cluster_centers_[cluster_class_index][0]+0.1, kmeans_object.cluster_centers_[cluster_class_index][1]))

        if  len(predicted_data) != 0:
            predicted_data_X_values = [predicted_data[i][0] for i in range(len(predicted_data))]
            predicted_data_Y_values = [predicted_data[i][1] for i in range(len(predicted_data))]
            plt.scatter(predicted_data_X_values , predicted_data_Y_values, s=200, marker="p", c="black",  label=f"predicted_test_data", zorder=10, alpha=0.7 )
            plt.title(f"K-means, k={cluster_size} - sub classes for class : {label} - predicted data")
        else:
            plt.title(f"K-means, k={cluster_size} - sub classes for class : {label}")

        l = plt.legend()
        l.set_zorder(50)  # put the legend on top
        #plt.show() 

    # Clean plot of data
    def plot_values_clean(self, images):
        plt.figure(figsize=(9, 9))
        images_bytes = [images[i].raw_bytes for i in range(len(images))]
        pca_images = PCA(n_components=2)
        pca_values_image = pca_images.fit_transform(images_bytes)
        pca_X_values = [pca_values_image[i][0] for i in range(len(pca_values_image))]
        pca_Y_values = [pca_values_image[i][1] for i in range(len(pca_values_image))]
        
        number_of_colors = 40
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        plt.scatter(pca_X_values , pca_Y_values, s=150, c=color[0],  label=f"clean-values-class : {images[0].label}", zorder=10, alpha=0.7 )
        
        plt.title(f"Clean plot image class : {images[0].label}")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top
        #plt.show() 

    # Taken from : https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    # teqniuque for finding optimal K-value
    def calculate_WSS(self, images, kmax):
        images_bytes = [images[i].raw_bytes for i in range(len(images))]
        pca_images = PCA(n_components=2)
        points = pca_images.fit_transform(images_bytes)
        sse = []
        for k in range(1, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

            sse.append(curr_sse)
        return sse
    
    # Plot the Elbow Graph
    def plot_WSS(self, wss_values, label):
        plt.figure(figsize=(9, 9))
        plt.plot(range(len(wss_values)), wss_values)
        plt.title(f"wss for class : {label}")


if __name__ == '__main__': 
    orl_image_loader = OrlDataLoader()
    image_viewer = ImageViewer()

    #Fetch training classes {2,3,5}
    images_class_2 = orl_image_loader.fetch_NSC_traning_set(2)
    images_class_3 = orl_image_loader.fetch_NSC_traning_set(3)
    images_class_5 = orl_image_loader.fetch_NSC_traning_set(5)

    #fetch testing classes {2,3,5}
    image_class_2_test = orl_image_loader.fetch_NSC_testing_set(2)
    image_class_3_test = orl_image_loader.fetch_NSC_testing_set(3)
    image_class_5_test = orl_image_loader.fetch_NSC_testing_set(5)

    # Init Model
    nearest_sub_class_classifier = NearestSubClassClassifier()

    # Calculate and plot WSS for class {2,3,5}
    wss_1 = nearest_sub_class_classifier.calculate_WSS(images_class_2, len(images_class_2))
    nearest_sub_class_classifier.plot_WSS(wss_1,2)

    wss_2 = nearest_sub_class_classifier.calculate_WSS(images_class_3, len(images_class_3))
    nearest_sub_class_classifier.plot_WSS(wss_2,3)
    
    wss_3 = nearest_sub_class_classifier.calculate_WSS(images_class_5, len(images_class_5))
    nearest_sub_class_classifier.plot_WSS(wss_3,5)

    # train the model for class k = 2 with m sub classes = 2
    #nearest_sub_class_classifier.plot_values_clean(images_class_2)
    kmeans_algoritm_results_2 = nearest_sub_class_classifier.find_sub_classes(images_class_2, 2)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_2[0],
                                                         kmeans_algoritm_results_2[1],
                                                         images_class_2[0].label,
                                                         kmeans_algoritm_results_2[2])    
    
    # train the model for class k = 3 with m sub classes = 2
    #nearest_sub_class_classifier.plot_values_clean(images_class_3)
    kmeans_algoritm_results_3 = nearest_sub_class_classifier.find_sub_classes(images_class_3, 2)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_3[0],
                                                         kmeans_algoritm_results_3[1],
                                                         images_class_3[0].label,
                                                         kmeans_algoritm_results_3[2])    
    
    # train the model for class k = 5 with m sub classes = 2
    #nearest_sub_class_classifier.plot_values_clean(images_class_5)
    kmeans_algoritm_results_5 = nearest_sub_class_classifier.find_sub_classes(images_class_5, 2)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_5[0],
                                                         kmeans_algoritm_results_5[1],
                                                         images_class_5[0].label,
                                                         kmeans_algoritm_results_5[2])    

    # Predict sub clases for class {2,3,5}
    testing_pca_2 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_2[0], image_class_2_test)
    testing_pca_3 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_3[0], image_class_3_test)
    testing_pca_5 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_5[0], image_class_5_test)

    #Plot sub classes for testing class 2
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_2[0],
                                                         kmeans_algoritm_results_2[1],
                                                         images_class_2[0].label,
                                                         kmeans_algoritm_results_2[2],
                                                         testing_pca_2 )   

    #Plot sub classes for testing class 3
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_3[0],
                                                         kmeans_algoritm_results_3[1],
                                                         images_class_3[0].label,
                                                         kmeans_algoritm_results_3[2],
                                                         testing_pca_3)

    #Plot sub classes for testing class 5
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_5[0],
                                                         kmeans_algoritm_results_5[1],
                                                         images_class_5[0].label,
                                                         kmeans_algoritm_results_5[2],
                                                         testing_pca_5)     
    plt.show()

                                                            
    
    
