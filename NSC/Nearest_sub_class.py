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

    def plot_sub_classes_kmeans(self, kmeans_object, pca_values_image, label, cluster_size):
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
                
        plt.title(f"K-means, k={cluster_size} - sub classes for class : {label}")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top
        #plt.show() 

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

        

if __name__ == '__main__': 
    orl_image_loader = OrlDataLoader()
    image_viewer = ImageViewer()

    images_class_2 = orl_image_loader.fetch_NSC_traning_set(2)
    images_class_3 = orl_image_loader.fetch_NSC_traning_set(3)
    images_class_5 = orl_image_loader.fetch_NSC_traning_set(5)


    nearest_sub_class_classifier = NearestSubClassClassifier()

    nearest_sub_class_classifier.plot_values_clean(images_class_2)
    kmeans_algoritm_results_2 = nearest_sub_class_classifier.find_sub_classes(images_class_2, 3)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_2[0], kmeans_algoritm_results_2[1], images_class_2[0].label, kmeans_algoritm_results_2[2])    
    
    nearest_sub_class_classifier.plot_values_clean(images_class_3)
    kmeans_algoritm_results_3 = nearest_sub_class_classifier.find_sub_classes(images_class_3, 3)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_3[0] , kmeans_algoritm_results_3[1], images_class_3[0].label, kmeans_algoritm_results_3[2])    
    
    nearest_sub_class_classifier.plot_values_clean(images_class_5)
    kmeans_algoritm_results_5 = nearest_sub_class_classifier.find_sub_classes(images_class_5, 3)
    nearest_sub_class_classifier.plot_sub_classes_kmeans(kmeans_algoritm_results_5[0] , kmeans_algoritm_results_5[1], images_class_5[0].label, kmeans_algoritm_results_5[2])    
    plt.show()



    
