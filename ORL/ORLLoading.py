import numpy as np
from PIL import Image

class OrlImage():
    """Data holder object for the supervised ORL data"""
    def __init__(self, raw_bytes, label):
        self.raw_bytes = raw_bytes
        self.label = label

class ImageViewer():
    def __init__(self):
        pass

    def show_image(self, raw_bytes, name):
        height, width = 40,30
        data = np.zeros((height, width, 3), dtype=np.uint8)
        scale_factor = 255

        image_pixel_count = 0
        for collum in range(width):
            for row in range(height):
                data[row][collum] = raw_bytes[image_pixel_count]*scale_factor
                image_pixel_count += 1
        
        img = Image.fromarray(data, "RGB")
        img.save(f"{name}.png")
        img.show()


class OrlDataLoader:
    def __init__(self):
        self._loaded_images = self._fetch_all_images()
        self._loaded_labels = self._fetch_all_labels()

    def _fetch_all_images(self):
        images_file = open("../ORl/orl_data.txt", "r")
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


    def _fetch_all_labels(self):
        labels_file = open("../ORL/orl_lbls.txt", "r")
        labels =  labels_file.read().split()
        labels_file.close()
        return labels


    def fetch_label_at_index(self, index):
        return self._loaded_labels[index]


    def fetch_image_at_index(self, image_index):
        image = self._loaded_images[:,image_index]
        return image    


    def fetch_NCC_traning_set(self):
        """ For each 10 images this function picks image 1-7 and skips image 8-9-10 --> 70% """
        traning_set = []

        pick_counter = 1
        for i in range(400):
            if(pick_counter <= 7):
                current_image = self.fetch_image_at_index(i)
                current_label = self.fetch_label_at_index(i)
                image_object = OrlImage(current_image, current_label)
                traning_set.append(image_object)
                pick_counter += 1
            else:
                if(pick_counter ==  10):
                    pick_counter = 0
                pick_counter += 1
        return traning_set


    def fetch_NCC_testing_set(self):
        testing_set = []

        pick_counter = 1
        for i in range(400):
            if (pick_counter <= 7):
                pick_counter += 1
            elif(pick_counter > 7 and pick_counter <= 10):
                current_image = self.fetch_image_at_index(i)
                current_label = self.fetch_label_at_index(i)
                image_object = OrlImage(current_image, current_label)
                testing_set.append(image_object)
                pick_counter +=1
            else:
                pick_counter = 2

        return testing_set  


    def fetch_NSC_traning_set(self, label):
        traning_set_NCC = self.fetch_NCC_traning_set()
        traning_set_NSC_sub = [traning_set_NCC[i] for i in range(len(self.fetch_NCC_traning_set())) if traning_set_NCC[i].label == str(label)]
        return traning_set_NSC_sub 