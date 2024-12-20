from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dispalcement_model import SwinTransformerSys  # 确保此模块路径正确
import matplotlib
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class ImagePatchDataset(Dataset):
    def __init__(self, def_image_path, ref_image_path, patch_size=128, stride=128, transform=None, region=None):
        self.def_image = Image.open(def_image_path).convert('L')
        self.ref_image = Image.open(ref_image_path).convert('L')

        if region is not None:
            self.def_image = self.def_image.crop(region)
            self.ref_image = self.ref_image.crop(region)

        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = self.extract_patches()

    def extract_patches(self):
        patches = []
        width, height = self.def_image.size
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                if x + self.patch_size > width:
                    x = width - self.patch_size
                if y + self.patch_size > height:
                    y = height - self.patch_size
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


def forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv, device='cuda', region=None):
    model.to(device)
    model.eval()

    if region is not None:
        region_width = region[2] - region[0]
        region_height = region[3] - region[1]
        disp_x_full = np.zeros((region_height, region_width))
        disp_y_full = np.zeros((region_height, region_width))
        count_full = np.zeros((region_height, region_width))
        left, upper = region[0], region[1]
    else:
        full_width, full_height = image_size  # 原始图像大小
        disp_x_full = np.zeros((full_height, full_width))
        disp_y_full = np.zeros((full_height, full_width))
        count_full = np.zeros((full_height, full_width))
        left, upper = 0, 0

    with torch.no_grad():
        for i, (def_patch, ref_patch, x, y) in enumerate(dataloader):
            def_patch = def_patch.to(device)
            ref_patch = ref_patch.to(device)

            output = model(def_patch, ref_patch)
            output = output.squeeze(0).cpu().numpy()

            patch_height, patch_width = output.shape[1], output.shape[2]

            for j in range(patch_height):
                for k in range(patch_width):
                    global_x = x.item() + k
                    global_y = y.item() + j

                    if 0 <= global_x < disp_x_full.shape[1] and 0 <= global_y < disp_x_full.shape[0]:
                        disp_x_full[global_y, global_x] += output[0, j, k]
                        disp_y_full[global_y, global_x] += output[1, j, k]
                        count_full[global_y, global_x] += 1

    count_full[count_full == 0] = 1e-5
    disp_x_full /= count_full
    disp_y_full /= count_full

    # 保存结果到CSV
    # with open(output_csv, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Index', 'X', 'Y', 'Disp_X', 'Disp_Y'])
    #
    #     index = 0
    #     for y_coord in range(disp_x_full.shape[0]):
    #         for x_coord in range(disp_x_full.shape[1]):
    #             writer.writerow([index, x_coord + left, y_coord + upper, disp_x_full[y_coord, x_coord], disp_y_full[y_coord, x_coord]])
    #             index += 1

    return disp_x_full, disp_y_full, count_full


def apply_filter(data_list, filter_type='gaussian', **kwargs):
    filtered_data_list = []
    for data in data_list:
        if filter_type == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            filtered_data = gaussian_filter(data, sigma=sigma)
        elif filter_type == 'median':
            size = kwargs.get('size', 3)
            filtered_data = median_filter(data, size=size)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        filtered_data_list.append(filtered_data)
    return filtered_data_list


# 叠加位移云图并添加内部颜色标尺
def overlay_displacement_on_image_with_internal_colorbar(original_image_path, disp_data, alpha=0.6, output_path=None, region=None, max_disp_value=5):
    font_path = 'C:\\Windows\\Fonts\\simhei.ttf'  # 请根据您的系统修改路径
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"指定的字体文件未找到: {font_path}")
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    original_image = Image.open(original_image_path).convert('RGB')
    original_image_np = np.array(original_image)

    disp_x, disp_y = disp_data
    disp_magnitude = np.sqrt(disp_x**2 + disp_y**2)

    if region is not None:
        left, upper, right, lower = region
        disp_full_magnitude = np.full((original_image_np.shape[0], original_image_np.shape[1]), np.nan)
        disp_full_magnitude[upper:lower, left:right] = disp_magnitude
    else:
        disp_full_magnitude = disp_magnitude

    norm = Normalize(vmin=0, vmax=max_disp_value)

    dpi = 100
    height, width, _ = original_image_np.shape
    figsize = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.imshow(original_image_np)
    mask = ~np.isnan(disp_full_magnitude)
    disp_masked = np.ma.masked_where(~mask, disp_full_magnitude)

    cmap = plt.cm.jet
    disp_overlay = ax.imshow(disp_masked, cmap=cmap, norm=norm, alpha=alpha, interpolation='nearest')

    ax.axis('off')

    if region is not None:
        left, upper, right, lower = region
        bar_width_pixels = 250
        bar_height_pixels = 650  # 增加颜色条的高度

        bar_width = bar_width_pixels / width
        bar_height = bar_height_pixels / height

        margin_pixels = 40
        x_position_pixels = right + margin_pixels
        y_position_pixels = upper + 200

        x_fraction = x_position_pixels / width
        y_fraction = y_position_pixels / height

        if x_position_pixels + bar_width_pixels > width:
            x_fraction = (left - margin_pixels - bar_width_pixels) / width
            if x_fraction < 0:
                x_fraction = 0
        if y_position_pixels + bar_height_pixels > height:
            y_fraction = (height - bar_height_pixels) / height

        cax = inset_axes(ax,
                         width=f"{bar_width*100}%",
                         height=f"{bar_height*100}%",  # 使用增加后的高度
                         loc='lower left',
                         bbox_to_anchor=(x_fraction, y_fraction-0.1, bar_width, bar_height),
                         bbox_transform=ax.transAxes,
                         borderpad=0)

        cax.set_facecolor('white')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        ticks = np.linspace(0, max_disp_value, num=5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{val:.2f}" for val in ticks])
        cbar.ax.tick_params(labelsize=12, colors='white')  # 设置刻度线和标签颜色为白色
        cbar.set_label('Displacement', fontsize=14, fontproperties=font_prop, color='white')  # 设置标签颜色为白色
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor('white')  # 设置边框颜色为白色

    else:
        bar_width_pixels = 70
        bar_height_pixels = 300  # 增加颜色条的高度
        bar_width = bar_width_pixels / width
        bar_height = bar_height_pixels / height
        x_fraction = 0.9
        y_fraction = 0.2

        cax = inset_axes(ax,
                         width=f"{bar_width*100}%",
                         height=f"{bar_height*100}%",  # 使用增加后的高度
                         loc='lower left',
                         bbox_to_anchor=(x_fraction, y_fraction, bar_width, bar_height),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        cax.set_facecolor('white')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        ticks = np.linspace(0, max_disp_value, num=5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{val:.2f}" for val in ticks])
        cbar.ax.tick_params(labelsize=14, colors='white')  # 设置刻度线和标签颜色为白色
        cbar.set_label('Displacement', fontsize=14, fontproperties=font_prop, color='white')  # 设置标签颜色为白色
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor('white')  # 设置边框颜色为白色

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    blended_image = Image.open(output_path)
    return blended_image


def main():
    try:

        # def_image_dir = r"E:\SwinT_UNET_data\Dan20240925\seris_bmp\right"
        # ref_image_path = r"E:\SwinT_UNET_data\Dan20240925\seris_bmp\right\series_step_460_right.bmp"
        def_image_dir = r"E:\SwinT_UNET_data\20240924_disp\x_disp\left\shift_right"
        ref_image_path = r"E:\SwinT_UNET_data\20240924_disp\x_disp\left\shift_right\shift_right_1.bmp"

        patch_size = 128
        stride = 60

        original_image = Image.open(ref_image_path)
        full_width, full_height = original_image.size
        image_size = (full_width, full_height)

        # define the region
        # region = (993, 90, 1249, 1570)  # (left, upper, right, lower)
        region = (650, 430, 1070, 790)  # (left, upper, right, lower)
        model = SwinTransformerSys()
        model_path = r"E:\SwinT_UNET_data\displacement_PTH\full_dataset_best_model_epoch_266.pth"
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        model.to('cuda')
        model.eval()

        def extract_serial_number(filename):
            match = re.search(r'step_(\d+)', filename)
            if match:
                return match.group(1)
            else:
                return None

        screenshot_dir = r"D:\OneDrive - ahu.edu.cn\--newstcpaper\images\displacement_tanslation_right"
        os.makedirs(screenshot_dir, exist_ok=True)

        # define the image index
        start_index = 1
        end_index = 20

        for current_index in range(start_index, end_index + 1):
            try:
                def_image_filename = f"shift_right_{current_index}.bmp"
                def_image_path = os.path.join(def_image_dir, def_image_filename)

                if not os.path.exists(def_image_path):
                    print(f"Image not found: {def_image_path}. Skipping.")
                    continue

                dataset = ImagePatchDataset(
                    def_image_path=def_image_path,
                    ref_image_path=ref_image_path,
                    patch_size=patch_size,
                    stride=stride,
                    transform=transform,
                    region=region
                )
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                def_serial_number = extract_serial_number(def_image_filename)
                ref_serial_number = extract_serial_number(os.path.basename(ref_image_path))

                output_csv = os.path.join(screenshot_dir, f"predicted_displacements_step_{current_index}.csv")

                disp_x_full, disp_y_full, count_full = forward_and_save_results(
                    model=model,
                    dataloader=dataloader,
                    image_size=image_size,
                    patch_size=patch_size,
                    stride=stride,
                    output_csv=output_csv,
                    device='cuda',
                    region=region
                )

                filtered_disp_x, filtered_disp_y = apply_filter(
                    data_list=[disp_x_full, disp_y_full],  # Correct parameter name
                    filter_type='gaussian',
                    sigma=2
                )
                # filtered_disp_x, filtered_disp_y = apply_filter(
                #     data_list=[disp_x_full, disp_y_full],  # Correct parameter name
                #     filter_type='median',
                #     size=3
                # )

                overlay_filename = f"overlay_displacement_step_{current_index}.png"
                overlay_output_path = os.path.join(screenshot_dir, overlay_filename)

                alpha = 0.5
                max_disp_value = 4

                combined_image = overlay_displacement_on_image_with_internal_colorbar(
                    original_image_path=def_image_path,
                    disp_data=np.array([filtered_disp_x, filtered_disp_y]),
                    alpha=alpha,
                    output_path=overlay_output_path,
                    region=region,
                    max_disp_value=max_disp_value
                )

                screenshot_filename = f"overlay_screenshot_step_{current_index}.png"
                screenshot_save_path = os.path.join(screenshot_dir, screenshot_filename)

                overlay_image = Image.open(overlay_output_path)

                expand_pixels = 200
                if region is not None:
                    left, upper, right, lower = region
                    expanded_left = max(left - expand_pixels, 0)
                    expanded_upper = max(upper - expand_pixels, 0)
                    expanded_right = min(right + expand_pixels, overlay_image.width)
                    expanded_lower = min(lower + expand_pixels, overlay_image.height)
                    expanded_region = (expanded_left, expanded_upper, expanded_right, expanded_lower)
                else:
                    expanded_region = (
                        max(0, 0 - expand_pixels),
                        max(0, 0 - expand_pixels),
                        min(overlay_image.width, full_width + expand_pixels),
                        min(overlay_image.height, full_height + expand_pixels)
                    )

                screenshot = overlay_image.crop(expanded_region)

                screenshot.save(screenshot_save_path)

                print(f"Processed step {current_index}:")
                print(f" - Displacement CSV: {output_csv}")
                print(f" - Overlay Image: {overlay_output_path}")
                print(f" - Screenshot: {screenshot_save_path}")

            except Exception as inner_e:
                print(f"An error occurred while processing step {current_index}: {inner_e}")

    except Exception as e:
        print(f"An error occurred in the main process: {e}")


if __name__ == '__main__':
    main()