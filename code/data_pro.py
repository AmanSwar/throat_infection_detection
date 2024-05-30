import torch
import torchvision
from torch import nn
from torchvision import transforms
import os
import PIL
from PIL import Image
from torch.utils.data import Dataset , Subset , DataLoader
import torchvision.models as models
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((256 , 256)) , 
    transforms.PILToTensor() , 
    transforms.ConvertImageDtype(torch.float32)
    ])

data_dir = "/home/aman/code/CV/throat_infection/data"

classes = sorted(os.listdir(data_dir))

class_to_label = {class_name: label for label ,class_name in enumerate(classes)}

class CustomImageDataset(Dataset):

    def __init__(self , data_dir , transforms=None):
        self.data_dir = data_dir
        self.transform = transforms
        self.image_path = []
        self.labels = []

        for class_name , label in class_to_label.items():
            class_dir = os.path.join(data_dir , class_name)

            for image_path in os.listdir(class_dir):
                self.image_path.append(os.path.join(class_dir , image_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self ,idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


dataset = CustomImageDataset(data_dir=data_dir , transforms=transform)

train_dataset = Subset(dataset , torch.arange(101))
valid_dataset = Subset(dataset , torch.arange(101 , len(dataset)))

device = torch.device("cuda")
batch_size = 5
train_dl = DataLoader(dataset , batch_size=batch_size , shuffle=True , pin_memory=True)
valid_dl = DataLoader(dataset , batch_size=batch_size , shuffle=True , pin_memory=True)

print('done')
