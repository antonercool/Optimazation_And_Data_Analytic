from mnist import MNIST
class MNISTImage():
    """Data holder object for the supervised ORL data"""
    def __init__(self, raw_bytes, label):
        self.raw_bytes = raw_bytes
        self.label = label


class MNISTDataLoader:
    def __init__(self):
        pass

    def fetch_traning_set(self):
        mndata = MNIST("C:/Projects/Optimazation_And_Data_Analytic/MNIST")

        images, labels = mndata.load_training()
           
        images_object_list = []
        for i in range(len(images)):
            image_object  = MNISTImage(images[i],labels[i])
            images_object_list.append(image_object)
            
        for image_index, images in enumerate(images_object_list):
            for byte_index, image_bytes in enumerate(images.raw_bytes):
                if image_bytes == 0:
                    continue
                else:
                    images_object_list[image_index].raw_bytes[byte_index] = image_bytes/255
                     
        return images_object_list
    
    def fetch_testing_set(self):
        mndata = MNIST("C:/Projects/Optimazation_And_Data_Analytic/MNIST")

        images, labels = mndata.load_testing()
        
        images_object_list = []
        for i in range(len(images)):
            image_object  = MNISTImage(images[i],labels[i])
            images_object_list.append(image_object)
        
        for image_index, images in enumerate(images_object_list):
            for byte_index, image_bytes in enumerate(images.raw_bytes):
                if image_bytes == 0:
                    continue
                else:
                    images_object_list[image_index].raw_bytes[byte_index] = image_bytes/255

        return images_object_list
    

    def fecth_testing_set_of_sub_class(self, sub_class):
        testing_set = self.fetch_testing_set() 
        return [testing_set[i] for i in range(len(testing_set)) if testing_set[i] == sub_class]


    def fecth_traning_set_of_sub_class(self, sub_class):
        testing_set = self.fetch_traning_set() 
        return [testing_set[i] for i in range(len(testing_set)) if testing_set[i] == sub_class]    