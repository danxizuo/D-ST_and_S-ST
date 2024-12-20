import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import scipy
# from V11_revised_Sobel_strain import SwinTransformerSys
from Swin_Unet_strain_22222 import SwinTransformerSys
import csv
from torch.cuda.amp import autocast, GradScaler
import h5py
# from V11_strain import SwinTransformerSys

class DeformationDataset(Dataset):
    def __init__(self, original_dir, transformed_dir, strain_dir, transform=None):
        self.original_dir = original_dir
        self.transformed_dir = transformed_dir
        self.strain_dir = strain_dir
        self.transform = transform
        # self.num_images = 85700
        self.num_images = 123700

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, f'reference{idx + 1}.bmp')
        transformed_image_path = os.path.join(self.transformed_dir, f'deformation{idx + 1}.bmp')
        strain_path = os.path.join(self.strain_dir, f'deformation{idx + 1}.mat')

        original_image = Image.open(original_image_path).convert('L')
        transformed_image = Image.open(transformed_image_path).convert('L')
        strain_data = scipy.io.loadmat(strain_path)

        strain = strain_data['E']  # 使用 'E' 键名
        strain = np.reshape(strain, (3, 128, 128))

        if self.transform:
            original_image = self.transform(original_image)
            transformed_image = self.transform(transformed_image)
        strain = torch.tensor(strain, dtype=torch.float32)

        return original_image, transformed_image, strain

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])


# Initialize the full dataset
dataset = DeformationDataset(
    original_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\int_noise\ref',
    transformed_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\int_noise\def',
    strain_dir=r'E:\CodeForPaper\Reset_VIT\----datasetDICNET\label\deformation',
    # original_dir=r'E:\SwinT_UNET_data\speckle pattern\dataset\reference_images',
    # transformed_dir=r'E:\SwinT_UNET_data\speckle pattern\dataset\deformed_images',
    # strain_dir=r'E:\SwinT_UNET_data\speckle pattern\dataset\strain_data',
    # original_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\ref',
    # transformed_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\def',
    # strain_dir=r'E:\SwinT_UNET_data\----datasetDICNET\label\deformation',
    transform=transform
)

# Split dataset into training and validation (90% train, 10% val)
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

generator = torch.Generator().manual_seed(42)

# Calculate sizes for training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# Create DataLoaders for training and validation
batch_size = 128
num_workers = 12

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Directory to save model checkpoints and logs
save_dir = r'F:\paper_pth\20241126_expanded_dataset_second'
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, 'cross_loss_log.csv')
scaler = GradScaler()

def train_with_scheduler(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, device='cuda', log_path='loss_log.csv'):
    model.to(device)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Open CSV file and write header
    with open(log_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        batch_count = 0

        for i, (original, transformed, strain) in enumerate(train_loader):
            # 将图像和标签移动到设备上
            original = original.to(device, non_blocking=True)
            transformed = transformed.to(device, non_blocking=True)
            strain = strain.to(device, non_blocking=True)

            # optimizer.zero_grad()
            # # 前向传播
            # outputs = model(original, transformed)
            # # 计算损失
            # loss = criterion(outputs, strain)
            optimizer.zero_grad()
            with autocast():
                outputs = model(original, transformed)
                loss = criterion(outputs, strain)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # 反向传播和优化
            # loss.backward()
            # optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_count += 1

            if batch_count == 100:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
                running_loss = 0.0
                batch_count = 0

        # Compute average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for original, transformed, strain in val_loader:
                original = original.to(device, non_blocking=True)
                transformed = transformed.to(device, non_blocking=True)
                strain = strain.to(device, non_blocking=True)

                # optimizer.zero_grad()
                # 前向传播
                outputs = model(original, transformed)
                # 计算损失
                loss = criterion(outputs, strain)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step()

        # Print epoch losses with full precision
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}')

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'---20241125_dataset_updated_strain_epoch_{epoch + 1}.pth'))

        # Append losses to CSV
        with open(log_path, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

    print('Training complete. Losses saved to', log_path)

# Initialize and run training
if __name__ == '__main__':
    model = SwinTransformerSys()
    # Optionally load a pre-trained model
    pretrained_path = r"F:\paper_pth\lr0001\---dataset_updated_strain_epoch_26.pth"
    model.load_state_dict(torch.load(pretrained_path), strict=False)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Define scheduler (adjust step_size and gamma as needed)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    # Start training
    train_with_scheduler(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=300,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_path=log_path
    )
