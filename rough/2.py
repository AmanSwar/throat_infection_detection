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

transform = transforms.Compose([transforms.PILToTensor()])
data_dir = "/home/aman/code/CV/throat_infection/data"

classes = sorted(os.listdir(data_dir))

class_to_label = {class_name: label for label ,class_name in enumerate(classes)}

class_to_label

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
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

        
dataset = CustomImageDataset(data_dir=data_dir , transforms=transform)

train_dataset = Subset(dataset , torch.arange(101))
valid_dataset = Subset(dataset , torch.arange(101 , len(dataset)))

device = torch.device("cuda")
batch_size = 10
train_dl = DataLoader(train_dataset , batch_size=batch_size , shuffle=True , pin_memory=True)
valid_dl = DataLoader(valid_dataset , batch_size=batch_size , shuffle=False , pin_memory=True)

from torchvision.models import resnet50 , ResNet50_Weights


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device=device)
in_feature = model.fc.in_features

model.fc = nn.Linear(in_feature , 1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters() , lr=0.001)

def train(model , n_epoch , train_dl , valid_dl , use_cuda=True):
    device = torch.device("cuda")

    loss_hist_train = [0] * n_epoch
    loss_hist_valid = [0] * n_epoch
    acc_hist_valid = [0] * n_epoch
    acc_hist_train = [0] * n_epoch


    # model.to(device)

    for epoch in range(n_epoch):
         model.train()

         for x, y in train_dl:
              
              x , y = x.to(device) , y.to(device)


              pred = model(x)
              loss = loss_fn(pred , y)
              loss.backward()
              optimizer.step()
              optimizer.zero_grad()
              loss_hist_train[epoch] += loss.item() * y.size(0)
              is_correct = (torch.argmax(pred, dim=1) == y).float()
              acc_hist_train[epoch] += is_correct.sum()


         loss_hist_train[epoch] /= len(train_dl.dataset)
         acc_hist_train[epoch] /= len(train_dl.dataset)

         model.eval()


         with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(device), y.to(device)  # Move data to the same device as the model

                pred = model(x)
                loss = loss_fn(pred, y)
                loss_hist_valid[epoch] += loss.item() * y.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y).float()
                acc_hist_valid[epoch] += is_correct.sum()

         loss_hist_valid[epoch] /= len(valid_dl.dataset)
         acc_hist_valid[epoch] /= len(valid_dl.dataset)


         print(f"epoch {epoch+1} accuracy "
              f"{acc_hist_train[epoch]:.4f} val_acuraccy : "
              f"{acc_hist_valid[epoch]:.4f}")
     
    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid

torch.manual_seed(1)

num_epoch = 20

hist = train(model , num_epoch , train_dl , valid_dl)