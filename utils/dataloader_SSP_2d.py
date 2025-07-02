from os.path import join
from os import listdir
import cv2
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder2D(data.Dataset):
    def __init__(self, file_dir, shape=(512, 512)):
        super(DatasetFromFolder2D, self).__init__()
        self.filenames = [x for x in listdir(file_dir) if is_image_file(x)]
        self.file_dir = file_dir
        self.shape = shape

    def __getitem__(self, index):
        img = cv2.imread(join(self.file_dir, self.filenames[index]), flags=cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.shape)
        img = img / 255.
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :]

        return img

    def __len__(self):
        return len(self.filenames)

