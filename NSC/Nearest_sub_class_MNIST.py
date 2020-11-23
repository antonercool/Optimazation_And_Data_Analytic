import sys
sys.path.append('../')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random as random
from MNIST.MNISTDataLoading import * 

class NearestSubClassClassifierMNIST:
    def __init__(self):
        pass

    def find_sub_classes(self, images, k_cluster_size):
        images_bytes = [images[i].raw_bytes for i in range(len(images))]
        pca_images = PCA(n_components=2)
        pca_values_image = pca_images.fit_transform(images_bytes)

        kmeans = KMeans(n_clusters=k_cluster_size, random_state=0)
        # image1,image2,image3, image4 ,image5 ,image6 ,image7 --> lies within class 2 (label 2)
        # [ 1      0      1       0        1         0    1]
        # some test image class 2   ---> Predict  == some SUB-class
        kmeans.fit(pca_values_image)
        return (kmeans, pca_values_image, k_cluster_size)


    def predict_sub_class(self, kmeans_object, testing_set):
        testing_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]
        pca_images = PCA(n_components=2)
        pca_testing_image = pca_images.fit_transform(testing_bytes)
        predicted_labels = kmeans_object.predict(pca_testing_image)

        print(f"Predicted data for label {testing_set[0].label} : {predicted_labels}");

        return predicted_labels

    def plot_prediction_data(self, k_kmeans_results, testing_set, predicted_labels, color):
        testing_bytes = [testing_set[i].raw_bytes for i in range(len(testing_set))]
        pca_images = PCA(n_components=2)
        pca_testing_image = pca_images.fit_transform(testing_bytes)

        kmeans_object, pca_values_image, cluster_size = k_kmeans_results
        plt.figure(figsize=(9, 9))
        pca_values_list = []
        for i in range(cluster_size): 
            temp_pca_value = []       
            for index, label_value in enumerate(predicted_labels):
                if label_value == i:
                    temp_pca_value.append(pca_testing_image[index]) 
            pca_values_list.append(temp_pca_value)        

        for cluster_class_index in range(cluster_size):
            current_pca_X_vals_class = [pca_values_list[cluster_class_index][i][0] for i in range(len(pca_values_list[cluster_class_index]))]
            current_pca_Y_vals_class = [pca_values_list[cluster_class_index][i][1] for i in range(len(pca_values_list[cluster_class_index]))]
           
            plt.scatter(current_pca_X_vals_class , current_pca_Y_vals_class, s=20, c=color[cluster_class_index],  label=f"sub-class-{cluster_class_index}", alpha=0.6 )
            plt.annotate(f"Centroid-sub-class-{cluster_class_index}", (kmeans_object.cluster_centers_[cluster_class_index][0]+0.4, kmeans_object.cluster_centers_[cluster_class_index][1]))

            plt.scatter(kmeans_object.cluster_centers_[cluster_class_index][0],kmeans_object.cluster_centers_[cluster_class_index][1], s=220 , marker="D", c='black')
            plt.scatter(kmeans_object.cluster_centers_[cluster_class_index][0],kmeans_object.cluster_centers_[cluster_class_index][1], s=150 , marker="D", c=color[cluster_class_index])
        
        plt.title(f"Prediction of test data to k={cluster_size} in class - {testing_set[0].label}")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top


    def plot_sub_classes_kmeans(self, k_kmeans_results, label, color):
        kmeans_object, pca_values_image, cluster_size = k_kmeans_results
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
           
            plt.scatter(current_pca_X_vals_class , current_pca_Y_vals_class, s=20, c=color[cluster_class_index],  label=f"sub-class-{cluster_class_index}", alpha=0.6 )
            plt.annotate(f"Centroid-class-{cluster_class_index}", (kmeans_object.cluster_centers_[cluster_class_index][0]+0.4, kmeans_object.cluster_centers_[cluster_class_index][1]))

            plt.scatter(kmeans_object.cluster_centers_[cluster_class_index][0],kmeans_object.cluster_centers_[cluster_class_index][1], s=220 , marker="D", c='black')
            plt.scatter(kmeans_object.cluster_centers_[cluster_class_index][0],kmeans_object.cluster_centers_[cluster_class_index][1], s=150 , marker="D", c=color[cluster_class_index])

        plt.title(f"Sub classes distribution for class = {label}, k-means = 4")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top

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

    # https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
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
    
    def plot_WSS(self, wss_values, label):
        plt.figure(figsize=(9, 9))
        plt.plot(range(1, len(wss_values)+1), wss_values)
        plt.title(f"wss for class : {label}")


    def calculate_metrics(self, predicted_labels, label):
        count_sub_class_0 = 0
        count_sub_class_1 = 0
        count_sub_class_2 = 0
        count_sub_class_3 = 0
        print(len(predicted_labels))
        for _, value in enumerate(predicted_labels):
            if value == 0:
                count_sub_class_0 +=1
            elif value == 1:
                count_sub_class_1 +=1
            elif value == 2:
                count_sub_class_2 +=1
            elif value == 3:
                count_sub_class_3 +=1
        print(f"printing metrics for class {label}")
        print(f"Probality of subclass - 0 {count_sub_class_0}/{len(predicted_labels)} = {count_sub_class_0/len(predicted_labels)}" )            
        print(f"Probality of subclass - 1 {count_sub_class_1}/{len(predicted_labels)} = {count_sub_class_1/len(predicted_labels)}" )            
        print(f"Probality of subclass - 2 {count_sub_class_2}/{len(predicted_labels)} = {count_sub_class_2/len(predicted_labels)}" )            
        print(f"Probality of subclass - 3 {count_sub_class_3}/{len(predicted_labels)} = {count_sub_class_3/len(predicted_labels)}" )            


## Takes about 30 seconds to run
if __name__ == '__main__': 
    MNISK_image_loader = MNISTDataLoader()

     #Fetch training classes {2,3,5}
    images_class_2 = MNISK_image_loader.fecth_traning_set_of_class(2)
    images_class_3 = MNISK_image_loader.fecth_traning_set_of_class(3)
    images_class_5 = MNISK_image_loader.fecth_traning_set_of_class(5)

    #fetch testing classes {2,3,5}
    image_class_2_test = MNISK_image_loader.fecth_testing_set_of_class(2)
    image_class_3_test = MNISK_image_loader.fecth_testing_set_of_class(3)
    image_class_5_test = MNISK_image_loader.fecth_testing_set_of_class(5)

    # Init Model
    nearest_sub_class_classifier = NearestSubClassClassifierMNIST()

    # Calculate and plot WSS for class {2,3,5}
    wss_1 = nearest_sub_class_classifier.calculate_WSS(images_class_2, 20 )
    nearest_sub_class_classifier.plot_WSS(wss_1,2)
    
    wss_2 = nearest_sub_class_classifier.calculate_WSS(images_class_3, 20)
    nearest_sub_class_classifier.plot_WSS(wss_2,3)
    
    wss_3 = nearest_sub_class_classifier.calculate_WSS(images_class_5, 20)
    nearest_sub_class_classifier.plot_WSS(wss_3,5)

    # Ploting different sizes of K from 2-5     --> 4 seems to be sweet point
    number_of_colors = 40
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    # train the model for class k = 2 with m sub classes = 4 and plot
    kmeans_algoritm_results_2_4 = nearest_sub_class_classifier.find_sub_classes(images_class_2, 4)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_2_4, images_class_2[0].label, color)
    
    # train the model for class k = 3 with m sub classes = 4 and plot
    kmeans_algoritm_results_3_4 = nearest_sub_class_classifier.find_sub_classes(images_class_3, 4)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_3_4, images_class_3[0].label, color)                                                            
    
    # train the model for class k = 5 with m sub classes = 4 and plot
    kmeans_algoritm_results_5_4 = nearest_sub_class_classifier.find_sub_classes(images_class_5, 4)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_5_4, images_class_5[0].label, color)
    
    # Predict subclasses for k = {2,3,5} and plot the results
    kmeans_algoritm_results_5_4 = nearest_sub_class_classifier.find_sub_classes(images_class_5, 4)
    prediction_labels5 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_5_4[0], image_class_5_test)
    prediction_labels2 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_2_4[0], image_class_2_test)
    prediction_labels3 = nearest_sub_class_classifier.predict_sub_class(kmeans_algoritm_results_3_4[0], image_class_3_test)
    nearest_sub_class_classifier.plot_prediction_data(kmeans_algoritm_results_5_4, image_class_5_test, prediction_labels5, color)

    #Calculate the metrics
    nearest_sub_class_classifier.calculate_metrics(prediction_labels2, 2)
    nearest_sub_class_classifier.calculate_metrics(prediction_labels3, 3)
    nearest_sub_class_classifier.calculate_metrics(prediction_labels5, 5)


    plt.show()


