import sys
sys.path.append('../')
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math 
import random as random
from MNIST.MNISTDataLoading import * 

class NearestClassCentroidMNIST:
    def __init__(self):
        pass  


    def train_nearest_class_centroid_model(self, traning_set):
        traning_data = [traning_set[i].raw_bytes for i in range(len(traning_set))]
        traning_labels = [traning_set[i].label for i in range(len(traning_set))] 

        pca_images = PCA(n_components=2)
        pca_training_image = pca_images.fit_transform(traning_data)

        nearest_class_centroid_model = NearestCentroid()
        # for each class calculate the mean of the class = centroid
        nearest_class_centroid_model.fit(pca_training_image, traning_labels)

        # return the traied model
        return nearest_class_centroid_model


    def apply_testing_set(self, testing_set, trained_model):
        testing_data = [testing_set[i].raw_bytes for i in range(len(testing_set))]

        pca_images = PCA(n_components=2)
        pca_testing_image = pca_images.fit_transform(testing_data)

        predicted_class = trained_model.predict(pca_testing_image)
        return predicted_class


    def evaluate_nearest_class_centroid_results(self, testing_set, predicted_labels):
        testing_labels = [testing_set[i].label for i in range(len(testing_set))]
        correct_predicted = 0
        for i in range(len(testing_labels)):
            if(testing_labels[i] == predicted_labels[i]):
                correct_predicted +=1

        return correct_predicted


    def calculate_closest_centroid(self, coordinate, centroids):
        smallest_dist = 2**31 -1

        current_centroid  = (0,0)
        for centroid in centroids:
            dist = math.sqrt((centroid[0] - coordinate[0])**2 + (centroid[1] - coordinate[1])**2)
            if dist < smallest_dist:
                smallest_dist = dist
                current_centroid = centroid
        return current_centroid   


    def plot_scatter_test_and_centroids_actual_prediction(self, test_images, ncc_object, predicted_labels, color):
        plt.figure(figsize=(9, 9))
        test_images_bytes = [test_images[i].raw_bytes for i in range(len(test_images)) ]

        pca_test = PCA(n_components=2)
        pca_test_values = pca_test.fit_transform(test_images_bytes)

        pca_X_vals_class_0 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 0 and test_images[i].label == 0]  
        pca_X_vals_class_1 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 1 and test_images[i].label == 1]  
        pca_X_vals_class_2 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 2 and test_images[i].label == 2]  
        pca_X_vals_class_3 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 3 and test_images[i].label == 3]  
        pca_X_vals_class_4 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 4 and test_images[i].label == 4]  
        pca_X_vals_class_5 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 5 and test_images[i].label == 5]  
        pca_X_vals_class_6 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 6 and test_images[i].label == 6]  
        pca_X_vals_class_7 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 7 and test_images[i].label == 7]  
        pca_X_vals_class_8 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 8 and test_images[i].label == 8]  
        pca_X_vals_class_9 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 9 and test_images[i].label == 9]  
        pca_X_vals_wrong   = [pca_test_values[i][0] for i in range(len(pca_test_values)) if pca_test_values[i][0] not in pca_X_vals_class_0 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_1 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_2 and       
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_3 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_4 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_5 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_6 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_7 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_8 and
                                                                                            pca_test_values[i][0] not in pca_X_vals_class_9]

        pca_Y_vals_class_0 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 0 and test_images[i].label == 0]
        pca_Y_vals_class_1 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 1 and test_images[i].label == 1]
        pca_Y_vals_class_2 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 2 and test_images[i].label == 2]
        pca_Y_vals_class_3 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 3 and test_images[i].label == 3]
        pca_Y_vals_class_4 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 4 and test_images[i].label == 4]
        pca_Y_vals_class_5 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 5 and test_images[i].label == 5]
        pca_Y_vals_class_6 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 6 and test_images[i].label == 6]
        pca_Y_vals_class_7 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 7 and test_images[i].label == 7]
        pca_Y_vals_class_8 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 8 and test_images[i].label == 8]
        pca_Y_vals_class_9 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 9 and test_images[i].label == 9]
        pca_Y_vals_wrong   = [pca_test_values[i][1] for i in range(len(pca_test_values)) if pca_test_values[i][1] not in pca_Y_vals_class_0 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_1 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_2 and       
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_3 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_4 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_5 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_6 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_7 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_8 and
                                                                                            pca_test_values[i][1] not in pca_Y_vals_class_9]


        plt.scatter(pca_X_vals_class_0 , pca_Y_vals_class_0, s=10, c=color[0])
        plt.scatter(pca_X_vals_class_1 , pca_Y_vals_class_1, s=10, c=color[1])
        plt.scatter(pca_X_vals_class_2 , pca_Y_vals_class_2, s=10, c=color[2])
        plt.scatter(pca_X_vals_class_3 , pca_Y_vals_class_3, s=10, c=color[3])
        plt.scatter(pca_X_vals_class_4 , pca_Y_vals_class_4, s=10, c=color[4])
        plt.scatter(pca_X_vals_class_5 , pca_Y_vals_class_5, s=10, c=color[5])
        plt.scatter(pca_X_vals_class_6 , pca_Y_vals_class_6, s=10, c=color[6])
        plt.scatter(pca_X_vals_class_7 , pca_Y_vals_class_7, s=10, c=color[7])
        plt.scatter(pca_X_vals_class_8 , pca_Y_vals_class_8, s=10, c=color[8])
        plt.scatter(pca_X_vals_class_9 , pca_Y_vals_class_9, s=10, c=color[9])
        plt.scatter(pca_X_vals_wrong, pca_Y_vals_wrong, s=10, color="black", alpha=0.2,label = "Wrongly predicted data")
        
        center_X_vals = [ncc_object.centroids_[i][0]  for i in range(len(ncc_object.centroids_))]
        center_Y_vals = [ncc_object.centroids_[i][1]  for i in range(len(ncc_object.centroids_))]
        for i in range(len(ncc_object.centroids_)):
            center_pca_X_value  = center_X_vals[i]
            center_pca_Y_value  = center_Y_vals[i]
            plt.scatter(center_pca_X_value, center_pca_Y_value, marker="D", s=220, c="black" )
            plt.scatter(center_pca_X_value, center_pca_Y_value, marker="D", s=180, c=color[i], label=f"centroid-{i}" )
        plt.title(f"Nearest Class Centroid MNIST data - Actual prediction of data")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top

    
    def plot_scatter_test_and_centroids(self, test_images, ncc_object, predicted_labels, color):
        plt.figure(figsize=(9, 9))
        test_images_bytes = [test_images[i].raw_bytes for i in range(len(test_images)) ]

        pca_test = PCA(n_components=2)
        pca_test_values = pca_test.fit_transform(test_images_bytes)

        pca_X_vals_class_0 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 0]  
        pca_X_vals_class_1 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 1]  
        pca_X_vals_class_2 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 2]  
        pca_X_vals_class_3 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 3]  
        pca_X_vals_class_4 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 4]  
        pca_X_vals_class_5 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 5]  
        pca_X_vals_class_6 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 6]  
        pca_X_vals_class_7 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 7]  
        pca_X_vals_class_8 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 8]  
        pca_X_vals_class_9 = [pca_test_values[i][0] for i in range(len(pca_test_values)) if predicted_labels[i] == 9]  
        pca_Y_vals_class_0 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 0]
        pca_Y_vals_class_1 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 1]
        pca_Y_vals_class_2 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 2]
        pca_Y_vals_class_3 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 3]
        pca_Y_vals_class_4 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 4]
        pca_Y_vals_class_5 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 5]
        pca_Y_vals_class_6 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 6]
        pca_Y_vals_class_7 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 7]
        pca_Y_vals_class_8 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 8]
        pca_Y_vals_class_9 = [pca_test_values[i][1] for i in range(len(pca_test_values)) if predicted_labels[i] == 9]

        plt.scatter(pca_X_vals_class_0 , pca_Y_vals_class_0, s=10, c=color[0], alpha=0.6 )
        plt.scatter(pca_X_vals_class_1 , pca_Y_vals_class_1, s=10, c=color[1], alpha=0.6 )
        plt.scatter(pca_X_vals_class_2 , pca_Y_vals_class_2, s=10, c=color[2], alpha=0.6 )
        plt.scatter(pca_X_vals_class_3 , pca_Y_vals_class_3, s=10, c=color[3], alpha=0.6 )
        plt.scatter(pca_X_vals_class_4 , pca_Y_vals_class_4, s=10, c=color[4], alpha=0.6 )
        plt.scatter(pca_X_vals_class_5 , pca_Y_vals_class_5, s=10, c=color[5], alpha=0.6 )
        plt.scatter(pca_X_vals_class_6 , pca_Y_vals_class_6, s=10, c=color[6], alpha=0.6 )
        plt.scatter(pca_X_vals_class_7 , pca_Y_vals_class_7, s=10, c=color[7], alpha=0.6 )
        plt.scatter(pca_X_vals_class_8 , pca_Y_vals_class_8, s=10, c=color[8], alpha=0.6 )
        plt.scatter(pca_X_vals_class_9 , pca_Y_vals_class_9, s=10, c=color[9], alpha=0.6)
        
        center_X_vals = [ncc_object.centroids_[i][0]  for i in range(len(ncc_object.centroids_))]
        center_Y_vals = [ncc_object.centroids_[i][1]  for i in range(len(ncc_object.centroids_))]
        for i in range(len(ncc_object.centroids_)):
            center_pca_X_value  = center_X_vals[i]
            center_pca_Y_value  = center_Y_vals[i]
            plt.scatter(center_pca_X_value, center_pca_Y_value, marker="D", s=220, c="black" )
            plt.scatter(center_pca_X_value, center_pca_Y_value, marker="D", s=180, c=color[i], label=f"centroid-{i}" )
        plt.title(f"Nearest Class Centroid MNIST data - Perfect prediction of data")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top

    
    def plot_scatter_test_dist_to_centroids_perfect_scenario(self, test_images, ncc_object, color):
        plt.figure(figsize=(9, 9))
        test_images_bytes = [test_images[i].raw_bytes for i in range(len(test_images)) ]

        pca_test = PCA(n_components=2)
        pca_test_values = pca_test.fit_transform(test_images_bytes)
                
        pca_X_vals_class = [pca_test_values[i][0] for i in range(len(pca_test_values))]
        pca_Y_vals_class = [pca_test_values[i][1] for i in range(len(pca_test_values))]
          
        center_X_vals = [ncc_object.centroids_[i][0]  for i in range(len(ncc_object.centroids_))]
        center_Y_vals = [ncc_object.centroids_[i][1]  for i in range(len(ncc_object.centroids_))]
        
        classses_X = [ [] for _ in range(10) ]
        classses_Y = [ [] for _ in range(10) ]
       
        for i in range(len(pca_X_vals_class)):
            closest_centroid = self.calculate_closest_centroid((pca_X_vals_class[i], pca_Y_vals_class[i]), ncc_object.centroids_)

            closest_centroid_index = [i for i in range(len(ncc_object.centroids_)) if  closest_centroid[0] == ncc_object.centroids_[i][0]][0]
            #for i,centroid in enumerate(pca_centroid_values):
            #    if closest_centroid[0] == centroid[0] and closest_centroid[1] == centroid[1]:
            classses_X[closest_centroid_index].append(pca_X_vals_class[i])
            classses_Y[closest_centroid_index].append(pca_Y_vals_class[i])
            #plt.plot([pca_X_vals_class[i], closest_centroid[0]], [pca_Y_vals_class[i], closest_centroid[1]] , linewidth=0.2, linestyle='--')
            #plt.annotate([pca_X_vals_class[i], closest_centroid[0]], [pca_Y_vals_class[i], closest_centroid[1]], "Smallest dist to centroid")
        for i in range(len(classses_X)):
            plt.scatter(classses_X[i] , classses_Y[i], s=10, c=color[i], alpha=0.6 )
            plt.scatter(center_X_vals[i] , center_Y_vals[i], s=230, c="black", marker="D")
            plt.scatter(center_X_vals[i] , center_Y_vals[i], s=180, c=color[i], marker="D", label=f"centroid-{i}" )

        plt.title(f"Nearest Class Centroid MNIST data - Perfect prediction to clostest centroid")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top


    def plot_scatter_how_to_predict(self, test_images, ncc_object, color):
        plt.figure(figsize=(9, 9))
        traning_images_bytes = [test_images[i].raw_bytes for i in range(len(test_images)) ]

        pca_test = PCA(n_components=2)
        pca_test_values = pca_test.fit_transform(traning_images_bytes)
                
        pca_X_vals_class = [pca_test_values[i][0] for i in range(len(pca_test_values))]
        pca_Y_vals_class = [pca_test_values[i][1] for i in range(len(pca_test_values))]
                  
        center_X_vals = [ncc_object.centroids_[i][0]  for i in range(len(ncc_object.centroids_))]
        center_Y_vals = [ncc_object.centroids_[i][1]  for i in range(len(ncc_object.centroids_))]
        
        classses_X = [ [] for _ in range(10) ]
        classses_Y = [ [] for _ in range(10) ]
       

        for i in range(len(pca_X_vals_class)):
            closest_centroid = self.calculate_closest_centroid((pca_X_vals_class[i], pca_Y_vals_class[i]), ncc_object.centroids_)

            closest_centroid_index = [i for i in range(len(ncc_object.centroids_)) if  closest_centroid[0] == ncc_object.centroids_[i][0]][0]

            classses_X[closest_centroid_index].append(pca_X_vals_class[i])
            classses_Y[closest_centroid_index].append(pca_Y_vals_class[i])
            plt.annotate(f"test-image{i}", xy=(pca_X_vals_class[i], pca_Y_vals_class[i]))
            plt.plot([pca_X_vals_class[i], closest_centroid[0]], [pca_Y_vals_class[i], closest_centroid[1]], c=color[closest_centroid_index], linewidth=0.2, linestyle='--')

        for i in range(len(classses_X)):
            plt.scatter(classses_X[i] , classses_Y[i], s=10, c=color[i], alpha=0.6 )
            plt.scatter(center_X_vals[i] , center_Y_vals[i], s=230, c="black", marker="D")
            plt.scatter(center_X_vals[i] , center_Y_vals[i], s=180, c=color[i], marker="D", label=f"centroid-{i}" )

        plt.title(f"Nearest Class Centroid MNIST data - How Nearest Class Centroid predicts")
        l = plt.legend()
        l.set_zorder(50)  # put the legend on top    
        
    

if __name__ == '__main__': 
    # Fetch traning/testing images
    data_loader = MNISTDataLoader()
    testing_set = data_loader.fetch_testing_set()
    traning_set = data_loader.fetch_traning_set()
    
    # Create class for traning/testing
    nearest_class_centroid_MNIST = NearestClassCentroidMNIST()
    ## Train model
    trained_model = nearest_class_centroid_MNIST.train_nearest_class_centroid_model(traning_set)
    ## Apply test data
    prediction_result = nearest_class_centroid_MNIST.apply_testing_set(testing_set, trained_model)
    ## Evaluate 
    correct_predicted = nearest_class_centroid_MNIST.evaluate_nearest_class_centroid_results(testing_set, prediction_result)

    number_of_colors = 10
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    nearest_class_centroid_MNIST.plot_scatter_test_and_centroids(testing_set, trained_model, prediction_result, colors)
    #nearest_class_centroid_MNIST.plot_scatter_test_dist_to_centroids_perfect_scenario(testing_set, trained_model, colors)
    nearest_class_centroid_MNIST.plot_scatter_how_to_predict(testing_set[0:10], trained_model, colors)
    nearest_class_centroid_MNIST.plot_scatter_test_and_centroids_actual_prediction(testing_set, trained_model, prediction_result, colors)

    print("correct predicted : ",correct_predicted , "/", len(testing_set))
    print("Pobability for prediction : " , correct_predicted/len(testing_set))
    plt.show()

  