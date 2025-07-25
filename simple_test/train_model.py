import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
# DataManager for loading data
from data_extraction.data_manager import DataManager
import matplotlib.pyplot as plt

class MRIAgeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        self.img_path = "/vol/miltank/projects/ukbb/data/whole_body/nifti_2d/"
        nifti_eids = [eid for eid in self.dataframe['eid'].unique() if os.path.exists(os.path.join(self.img_path, f"{eid}.png"))]
        self.dataframe = self.dataframe[self.dataframe['eid'].isin(nifti_eids)]
        self.dataframe['img_path'] = self.dataframe['eid'].apply(lambda x: os.path.join(self.img_path, f"{x}.png"))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        age = self.dataframe.iloc[idx]['age']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)



def train_model(data_mgr):
    print("starting training")
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
    ])
    data_manager = data_mgr
    print("DataManager initialized")    
    train_frame = data_manager.data[0]
    train_set = MRIAgeDataset(train_frame, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    print(f"Train loader loaded, size of dataset: {len(train_set)}")
    val_frame = data_manager.data[1]
    val_set = MRIAgeDataset(val_frame, transform=transform)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    print(f"Validation loader loaded, size of dataset: {len(val_set)}")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model initialized and moved to device {device}")
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    model.train()
    print("Starting training loop")
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        for inputs, ages in train_loader:
            inputs, ages = inputs.to(device), ages.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        for inputs, ages in val_loader:
            inputs, ages = inputs.to(device), ages.to(device).unsqueeze(1)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, ages)
                running_val_loss += loss.item() * inputs.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Training complete")
    
    # Save the model
    model_save_path = "mri_age_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # plot
    plt.plot(range(num_epochs), losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xticks(range(num_epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig("training_loss_plot.png")
    plt.close()
    print("Training loss plot saved as training_loss_plot.png")

def test_model(data_mgr):
    # load model
    print("Starting testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("mri_age_model.pth"))
    model = model.to(device)
    model.eval()
    test_frame = data_mgr.data[1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                             [0.229, 0.224, 0.225])  # ImageNet stds
    ])
    test_set = MRIAgeDataset(test_frame, transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test loader loaded, size of dataset: {len(test_set)}")
    criterion = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, ages in test_loader:
            inputs, ages = inputs.to(device), ages.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, ages)
            total_loss += loss.item() * inputs.size(0)
    average_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss: {average_loss:.4f}")
    # plot 

    plt.plot(range(len(test_loader)), [average_loss] * len(test_loader), label='Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Test Loss per Batch, total loss: {average_loss:.4f}')
    plt.legend()
    plt.savefig("test_loss_plot.png")
    plt.close()
    print("Test loss plot saved as test_loss_plot.png")


if __name__ == "__main__":
    print("STARTING TRAINING")
    data_mgr = DataManager("regression")
    print("DataManager initialized")
    train_model(data_mgr)
    test_model(data_mgr)

    print("FINISHED")
