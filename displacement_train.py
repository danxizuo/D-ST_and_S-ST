import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import scipy.io as sio
import scipy
from timm.models.swin_transformer import SwinTransformer
import random
import csv  # For saving loss history to CSV
from Swin_Unet_disp_22222 import SwinTransformerSys
# 设置全局随机种子以确保完全可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class DeformationDataset(Dataset):
    def __init__(self, original_dir, transformed_dir, displacement_dir, transform=None):
        self.original_dir = original_dir
        self.transformed_dir = transformed_dir
        self.displacement_dir = displacement_dir
        self.transform = transform
        # self.num_images = 96580
        # self.num_images = 85700
        # self.num_images = 123700
        self.num_images = 161700
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, f'reference{idx + 1}.bmp')
        transformed_image_path = os.path.join(self.transformed_dir, f'deformation{idx + 1}.bmp')
        displacement_path = os.path.join(self.displacement_dir, f'displacement{idx + 1}.mat')

        original_image = Image.open(original_image_path).convert('L')
        transformed_image = Image.open(transformed_image_path).convert('L')
        displacement_data = scipy.io.loadmat(displacement_path)

        displacement = displacement_data['uu']  # 使用 'uu' 键名
        displacement = np.reshape(displacement, (2, 128, 128))

        if self.transform:
            original_image = self.transform(original_image)
            transformed_image = self.transform(transformed_image)
        displacement = torch.tensor(displacement, dtype=torch.float32)

        return original_image, transformed_image, displacement

# Enhanced transform with data augmentation (optional)
transform = transforms.Compose([

    transforms.ToTensor()
    # transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize if necessary
    # Add more augmentations here if desired, e.g., RandomHorizontalFlip
])

# Initialize the full dataset
full_dataset = DeformationDataset(
    original_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\int_noise\ref',
    transformed_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\int_noise\def',
    displacement_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\label\displacement',
    transform=transform
)

# Calculate split sizes
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 设置固定的随机种子以确保数据集划分的一致性
seed = 42
generator = torch.Generator().manual_seed(seed)

# Split the dataset with the fixed random seed
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

# Data loaders
batch_size = 64
num_workers = 14

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

save_dir = r'F:\paper_pth\22222_disp'
os.makedirs(save_dir, exist_ok=True)

def train_with_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=15,
                         device='cuda'):
    model.to(device)
    best_val_loss = float('inf')

    # Path to the CSV file
    loss_history_path = os.path.join(save_dir, 'loss_history.csv')

    # Initialize the CSV file with headers if it doesn't exist
    if not os.path.exists(loss_history_path):
        with open(loss_history_path, mode='w', newline='') as csv_file:
            fieldnames = ['epoch', 'train_loss', 'val_loss']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        batch_count = 0

        for i, (original, transformed, displacement) in enumerate(train_loader):
            original, transformed, displacement = original.to(device), transformed.to(device), displacement.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(original, transformed)

            # Compute loss
            loss = criterion(outputs, displacement)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_count += 1

            if batch_count == 100:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
                running_loss = 0.0
                batch_count = 0

        # Scheduler step
        scheduler.step()

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for original, transformed, displacement in test_loader:
                original, transformed, displacement = original.to(device), transformed.to(device), displacement.to(
                    device)
                outputs = model(original, transformed)
                loss = criterion(outputs, displacement)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}')

        # Append the losses to the CSV file
        with open(loss_history_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'246_best_model_epoch_{epoch + 1}.pth'))
            print(f'Best model saved at epoch {epoch + 1}')

        # Optionally, implement early stopping here based on validation loss

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f'246_V11_train_test_{epoch + 1}.pth'))

    # Optionally, return the loss history
    # return loss_history

if __name__ == '__main__':
    model = SwinTransformerSys()
    # pretrained_model_path = r"F:\displacement_pth\246_best_model_epoch_42.pth"
    # if os.path.exists(pretrained_model_path):
    #     model.load_state_dict(torch.load(pretrained_model_path))
    #     print("Pre-trained model loaded successfully.")
    # else:
    #     print("Pre-trained model not found. Training from scratch.")

    criterion = nn.MSELoss()

    # Updated optimizer with weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Adjust weight_decay as needed

    # Learning rate scheduler: Reduce LR on plateau can be beneficial
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # Set the number of epochs appropriately
    num_epochs = 300  # Adjust as needed

    train_with_scheduler(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
