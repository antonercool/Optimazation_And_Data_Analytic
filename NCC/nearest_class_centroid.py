from PIL import Image
from sklearn.neighbors import NearestCentroid
import numpy as np

def fetch_all_images():
    images_file = open("../ORL_txt/orl_data.txt", "r")
    image_lines = images_file.readlines()
    images_file.close()
    
    images_matrix = np.zeros((1200,400),  dtype=np.float)
    row = 0
    collum = 0
    for line in image_lines:
        for pixel in line.split():
            images_matrix[row][collum] = float(pixel)
            collum = collum + 1
        collum = 0
        row = row + 1
    return images_matrix                


def fetch_all_labels():
    labels_file = open("../ORL_txt/orl_lbls.txt", "r")
    labels =  labels_file.read().split()
    labels_file.close()
    return labels

def fetch_label_at_index(labels, index):
    return labels[index]


def fetch_image_at_index(images, image_index):
    image = images[:,image_index]
    return image    


def image_show(image):
    height, width = 40,30
    data = np.zeros((height, width, 3), dtype=np.uint8)
    scale_factor = 255

    image_pixel_count = 0
    for collum in range(width):
        for row in range(height):
            data[row][collum] = image[image_pixel_count]*scale_factor
            image_pixel_count += 1
    
    img = Image.fromarray(data, "RGB")
    img.show()


def fetch_traning_set():
    image_all = fetch_all_images()
    label_all = fetch_all_labels()
    traning_set = []

    pick_counter = 1
    for i in range(400):
        if(pick_counter <= 7):
            current_image = fetch_image_at_index(image_all, i)
            current_label = fetch_label_at_index(label_all, i)
            traning_set.append((current_image,current_label))
            pick_counter += 1
        else:
            if(pick_counter ==  10):
                pick_counter = 0
            pick_counter += 1
    return traning_set


def fetch_testing_set():
    image_all = fetch_all_images()
    label_all = fetch_all_labels()
    traning_set = []

    pick_counter = 1
    for i in range(400):
        if (pick_counter <= 7):
            pick_counter += 1
        elif(pick_counter > 7 and pick_counter <= 10):
            current_image = fetch_image_at_index(image_all, i)
            current_label = fetch_label_at_index(label_all, i)
            traning_set.append((current_image,current_label))
            pick_counter +=1
        else:
            pick_counter = 2

    return traning_set  


def train_nearest_class_centroid_model(traning_set):
    traning_data = [traning_set[i][0] for i in range(len(traning_set))]
    traning_labels = [traning_set[i][1] for i in range(len(traning_set))] 

    nearest_class_centroid_model = NearestCentroid()
    # for each class calculate the mean of the class = centroid
    nearest_class_centroid_model.fit(traning_data, traning_labels)

    # return the traied model
    return nearest_class_centroid_model

def evaluate_class_for_image(testing_set, trained_model):
    testing_data = [testing_set[i][0] for i in range(len(testing_set))]
    predicted_class = trained_model.predict(testing_data)
    return predicted_class



if __name__ == '__main__': 
    image_all = fetch_all_images()
    #image_1 = fetch_image_at_index(image_all, 10)
    #image_show(image_1)

    label_all = fetch_all_labels()
    #label_1 = fetch_label_at_index(label_all, 9)
    #print(label_1)

    traning_set = fetch_traning_set()
    
    testing_set = fetch_testing_set()
    testing_labels = [testing_set[i][1] for i in range(len(testing_set))] 

    trained_model = train_nearest_class_centroid_model(traning_set)

    result = evaluate_class_for_image(testing_set, trained_model)

    print(result)

    correct_predicted = 0
    
    for i in range(len(testing_labels)):
        if(testing_labels[i] == result[i]):
            correct_predicted +=1

    #print("label known :  \n", testing_labels , "\n label predicted : \n", result)
    print("correct predicted : ",correct_predicted , "/", len(testing_labels))
    print("Pobability for prediction : " , correct_predicted/len(testing_labels))
    

   