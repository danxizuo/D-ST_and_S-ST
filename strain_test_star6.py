import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from strain_model import SwinTransformerSys  # Ensure this module is accessible


def forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_folder, device='cuda',
                             discard_pixels=2):
    """
    Processes image patches through the model and saves strain data into CSV files.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        image_size (tuple): (width, height) of the original images.
        patch_size (int): Size of each image patch.
        stride (int): Stride for patch extraction.
        output_folder (str): Directory to save the CSV files.
        device (str): Device to run the model on ('cuda' or 'cpu').
        discard_pixels (int): Number of pixels to discard at the borders of patches to avoid overlap artifacts.

    Returns:
        tuple: strain_x, strain_y, strain_xy matrices.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    model.to(device)
    model.eval()

    width, height = image_size
    strain_x = np.zeros((height, width))
    strain_y = np.zeros((height, width))
    strain_xy = np.zeros((height, width))
    count = np.zeros((height, width))

    with torch.no_grad():
        for i, (def_patch, ref_patch, x, y) in enumerate(dataloader):
            def_patch = def_patch.to(device)
            ref_patch = ref_patch.to(device)

            output = model(def_patch, ref_patch)
            output = output.squeeze(0).cpu().numpy()

            for j in range(patch_size):
                for k in range(patch_size):
                    global_x = x + k
                    global_y = y + j

                    # Check if the current patch pixel is within the discard zone
                    if (x == 0 and k < discard_pixels) or (y == 0 and j < discard_pixels) or \
                            (x + patch_size >= width and k >= patch_size - discard_pixels) or \
                            (y + patch_size >= height and j >= patch_size - discard_pixels):
                        # Always include border pixels
                        include_pixel = True
                    else:
                        # Include only central pixels
                        if (discard_pixels <= j < patch_size - discard_pixels) and (
                                discard_pixels <= k < patch_size - discard_pixels):
                            include_pixel = True
                        else:
                            include_pixel = False

                    if include_pixel:
                        # Ensure indices are within image bounds
                        if 0 <= global_x < width and 0 <= global_y < height:
                            strain_x[global_y, global_x] += output[0, j, k]
                            strain_y[global_y, global_x] += output[1, j, k]
                            strain_xy[global_y, global_x] += output[2, j, k]
                            count[global_y, global_x] += 1

    # Avoid division by zero
    count[count == 0] = 1e-5
    strain_x /= count
    strain_y /= count
    strain_xy /= count

    # Define output file paths
    # output_csv_x = os.path.join(output_folder, 'strain_x.csv')
    # output_csv_y = os.path.join(output_folder, 'strain_y.csv')
    # output_csv_xy = os.path.join(output_folder, 'strain_xy.csv')
    #
    # # Save the strain data to CSV files
    # np.savetxt(output_csv_x, strain_x, delimiter=',')
    # np.savetxt(output_csv_y, strain_y, delimiter=',')
    # np.savetxt(output_csv_xy, strain_xy, delimiter=',')

    return strain_x, strain_y, strain_xy


class ImagePatchDataset(Dataset):
    """
    Custom Dataset for extracting image patches from deformed and reference images.
    """

    def __init__(self, def_image_path, ref_image_path, patch_size=128, stride=120, transform=None):
        self.def_image = Image.open(def_image_path).convert('L')
        self.ref_image = Image.open(ref_image_path).convert('L')
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = self.extract_patches()

    def extract_patches(self):
        patches = []
        width, height = self.def_image.size
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Adjust patch position to stay within image bounds
                if x + self.patch_size > width:
                    x = width - self.patch_size
                if y + self.patch_size > height:
                    y = height - self.patch_size
                # Crop patches from both images
                def_patch = self.def_image.crop((x, y, x + self.patch_size, y + self.patch_size))
                ref_patch = self.ref_image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append((def_patch, ref_patch, x, y))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        def_patch, ref_patch, x, y = self.patches[idx]
        if self.transform:
            def_patch = self.transform(def_patch)
            ref_patch = self.transform(ref_patch)
        return def_patch, ref_patch, x, y


transform = transforms.Compose([
    transforms.ToTensor()
])


def plot_displacement(disp_x, disp_y, image_size, title):
    """
    Plots the displacement in the Y direction with normalization.

    Args:
        disp_x (np.ndarray): Displacement in X direction.
        disp_y (np.ndarray): Displacement in Y direction.
        image_size (tuple): (width, height) of the image.
        title (str): Title for the plot.
    """
    disp_error = disp_y
    disp_error_normalized = np.clip(disp_error, -0.5, 0.5)

    fig, ax = plt.subplots(figsize=(40, 5))
    cax = ax.imshow(disp_error_normalized, cmap='jet', vmin=-0.05, vmax=0.05, aspect='auto')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.show()


if __name__ == '__main__':
    # Define paths to the deformed and reference images
    def_image_path = r'D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star6StrainNoisy\Star6StrainNoisy\DIC_Challenge_Star_Strain_Noise_Def.tif'
    ref_image_path = r'D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star6StrainNoisy\Star6StrainNoisy\DIC_Challenge_Star_Strain_Noise_Ref.tif'
    # def_image_path = "D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star3NoNoiseStrain\DIC_Challenge_Wave_Deformed_Strain.tif"
    # ref_image_path = "D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star3NoNoiseStrain\DIC_Challenge_Wave_Reference_Strain.tif"
    # def_image_path = "E:\SwinT_UNET_data\strain_images_cropped\series_step_182_right.bmp"
    # ref_image_path = "E:\SwinT_UNET_data\strain_images_cropped\series_step_200_right.bmp"
    # Parameters
    patch_size = 128
    stride = 100
    image_size = (4000, 501)  # (width, height)

    # Define output folder
    output_folder = r'D:\OneDrive - ahu.edu.cn\--newstcpaper\data\strain_STDIC_multi'  # Change this path as needed

    # Create dataset and dataloader
    dataset = ImagePatchDataset(def_image_path, ref_image_path, patch_size, stride, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize and load the model
    model = SwinTransformerSys()
    model_path = r"F:\paper_pth\new_dataset\dataset_updated_strain_epoch_296.pth"  # Update this path if necessary
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # Forward pass and save results
    disp_x, disp_y, strain_xy = forward_and_save_results(
        model, dataloader, image_size, patch_size, stride, output_folder,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Plot displacement in Y direction
    plot_displacement(disp_x, disp_y, image_size, title=os.path.basename(model_path))

    print(f'Results saved to {output_folder}')
