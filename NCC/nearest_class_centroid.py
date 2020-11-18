import sys
sys.path.append('../')
from ORL.ORLLoading import * 
from sklearn.neighbors import NearestCentroid

def train_nearest_class_centroid_model(traning_set):
    traning_data = [traning_set[i].raw_bytes for i in range(len(traning_set))]
    traning_labels = [traning_set[i].label for i in range(len(traning_set))] 

    nearest_class_centroid_model = NearestCentroid()
    # for each class calculate the mean of the class = centroid
    nearest_class_centroid_model.fit(traning_data, traning_labels)

    # return the traied model
    return nearest_class_centroid_model

def apply_testing_set(testing_set, trained_model):
    testing_data = [testing_set[i].raw_bytes for i in range(len(testing_set))]
    predicted_class = trained_model.predict(testing_data)
    return predicted_class

def evaluate_nearest_class_centroid_results(testing_labels, predicted_labels):
    correct_predicted = 0
    for i in range(len(testing_labels)):
        if(testing_labels[i] == predicted_labels[i]):
            correct_predicted +=1

    return correct_predicted

if __name__ == '__main__': 
    """Prereq install:
        - 'PIL' for image displaying
        - 'sklearn' for the machine learing part """
    
    orl_image_loader = OrlDataLoader()
    image_viewer = ImageViewer()

    #image_1 = orl_image_loader.fetch_image_at_index(1)
    #image_viewer.show_image(image_1)

    traning_set = orl_image_loader.fetch_NCC_traning_set()
    testing_set = orl_image_loader.fetch_NCC_testing_set()

    testing_labels = [testing_set[i].label for i in range(len(testing_set))] 
    trained_model = train_nearest_class_centroid_model(traning_set)
    testing_results = apply_testing_set(testing_set, trained_model)

    print(f"predicted labels : \n {testing_results}")
    print(f"known test labels : \n {np.array(testing_labels)}")
    correct_predicted = evaluate_nearest_class_centroid_results(testing_labels, testing_results)

    print("correct predicted : ",correct_predicted , "/", len(testing_labels))
    print("Pobability for prediction : " , correct_predicted/len(testing_labels))
    

   