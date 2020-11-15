from PIL import Image
import numpy as np

def fetch_all_images():
    images_file = open("..\ORL_txt\orl_data.txt", "r")
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
    labels_file = open("..\ORL_txt\orl_lbls.txt", "r")
    labels =  labels_file.read().split()
    print(labels)
    

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
    img.save('my.png')
    img.show()


if __name__ == '__main__': 
    images = fetch_all_images()
    image1 = fetch_image_at_index(images, 10)
    image_show(image1)
    #fetch_all_labels()