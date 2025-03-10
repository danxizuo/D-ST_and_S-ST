import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape

    # Pad the input if needed
    # pad_h = (window_size - H % window_size) % window_size
    # pad_w = (window_size - W % window_size) % window_size
    # if pad_h > 0 or pad_w > 0:
    #     # x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    # 将 x 从 (B, H, W, C) 转换为 (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    # 计算填充
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    # 应用填充
    x = F.pad(x, (0, pad_w, 0, pad_h))
    # 重新调整维度回原来的顺序
    x = x.permute(0, 2, 3, 1)

    # Compute number of windows
    H_padded, W_padded = H + pad_h, W + pad_w
    num_windows_h = H_padded // window_size
    num_windows_w = W_padded // window_size

    # Reshape and permute
    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (H, W, H_padded, W_padded)


def window_reverse(windows, window_size, H_padded, W_padded, H, W):
    B = int(windows.shape[0] / ((H_padded // window_size) * (W_padded // window_size)))

    # Restore to the padded shape
    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)

    # Remove padding
    x = x[:, :H, :W, :].contiguous()

    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Wh, Ww]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # Check input dimensions
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor as input, got {x.dim()}D tensor")

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, \
            "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - \
                mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows, (_, _, H_padded, W_padded) = window_partition(
            shifted_x, self.window_size)

        # Reshape to 3D
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H_padded, W_padded, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, \
            f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale,
            c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class BasicLayerUp(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch expanding layer
        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=2, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size  # (H, W)
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution  # [H_patch, W_patch]
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size[0] - H %
                 self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W %
                 self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))  # Padding height and width
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


# class StrainLayer(nn.Module):
#     def __init__(self):
#         super(StrainLayer, self).__init__()
#         # 定义中心差分核
#         kernel_dx = torch.tensor([[0, 0, 0],
#                                   [-0.5, 0, 0.5],
#                                   [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
#         kernel_dy = torch.tensor([[0, -0.5, 0],
#                                   [0, 0, 0],
#                                   [0, 0.5, 0]], dtype=torch.float32).view(1, 1, 3, 3)
#         # 将核注册为缓冲区
#         self.register_buffer('kernel_dx', kernel_dx)
#         self.register_buffer('kernel_dy', kernel_dy)
#
#     def forward(self, displacement):
#         disp_x = displacement[:, 0:1, :, :]  # B x 1 x H x W
#         disp_y = displacement[:, 1:2, :, :]  # B x 1 x H x W
#
#         # 手动进行反射填充
#         padding = 1  # 卷积核大小为 3，故 padding 为 1
#         disp_x_padded = F.pad(disp_x, (padding, padding, padding, padding), mode='reflect')
#         disp_y_padded = F.pad(disp_y, (padding, padding, padding, padding), mode='reflect')
#
#         # 计算位移梯度
#         grad_disp_x_x = F.conv2d(disp_x_padded, self.kernel_dx)
#         grad_disp_x_y = F.conv2d(disp_x_padded, self.kernel_dy)
#         grad_disp_y_x = F.conv2d(disp_y_padded, self.kernel_dx)
#         grad_disp_y_y = F.conv2d(disp_y_padded, self.kernel_dy)
#
#         # 计算应变分量
#         strain_xx = grad_disp_x_x
#         strain_yy = grad_disp_y_y
#         strain_xy = 0.5 * (grad_disp_x_y + grad_disp_y_x)
#
#         return torch.cat([strain_xx, strain_yy, strain_xy], dim=1)


class StrainLayer(nn.Module):
    def __init__(self):
        super(StrainLayer, self).__init__()
        # 定义 Sobel 核，并注册为缓冲区
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, displacement):
        disp_x = displacement[:, 0:1, :, :]  # B x 1 x H x W
        disp_y = displacement[:, 1:2, :, :]
        # 计算梯度
        strain_xx = F.conv2d(disp_x, self.sobel_x, padding=1)
        strain_yy = F.conv2d(disp_y, self.sobel_y, padding=1)
        strain_xy = 0.5 * (F.conv2d(disp_y, self.sobel_x, padding=1) +
                           F.conv2d(disp_x, self.sobel_y, padding=1))
        return torch.cat([strain_xx, strain_yy, strain_xy], dim=1)


# class StrainLayer(nn.Module):
#     def __init__(self):
#         super(StrainLayer, self).__init__()
#         # 使用可学习的卷积层
#         self.conv_xx = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         self.conv_yy = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         self.conv_xy = nn.Conv2d(2, 1, kernel_size=3, padding=1)
#
#     def forward(self, displacement):
#         disp_x = displacement[:, 0:1, :, :]  # B x 1 x H x W
#         disp_y = displacement[:, 1:2, :, :]  # B x 1 x H x W
#
#         # 计算应变分量
#         strain_xx = self.conv_xx(disp_x)
#         strain_yy = self.conv_yy(disp_y)
#         strain_xy_input = torch.cat([disp_x, disp_y], dim=1)
#         strain_xy = self.conv_xy(strain_xy_input)
#
#         return torch.cat([strain_xx, strain_yy, strain_xy], dim=1)
# class StrainLayer(nn.Module):
#     def __init__(self):
#         super(StrainLayer, self).__init__()
#         # 定义固定核
#         central_diff_kernel_x = torch.tensor([[0, 0, 0],
#                                               [-0.5, 0, 0.5],
#                                               [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
#         central_diff_kernel_y = torch.tensor([[0, -0.5, 0],
#                                               [0, 0, 0],
#                                               [0, 0.5, 0]], dtype=torch.float32).view(1, 1, 3, 3)
#         sobel_kernel_x = torch.tensor([[-1, 0, 1],
#                                        [-2, 0, 2],
#                                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         sobel_kernel_y = torch.tensor([[-1, -2, -1],
#                                        [0, 0, 0],
#                                        [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         # 注册核为缓冲区
#         self.register_buffer('central_diff_kernel_x', central_diff_kernel_x)
#         self.register_buffer('central_diff_kernel_y', central_diff_kernel_y)
#         self.register_buffer('sobel_kernel_x', sobel_kernel_x)
#         self.register_buffer('sobel_kernel_y', sobel_kernel_y)
#
#         # 用于组合输出的可学习权重
#         self.weight_xx = nn.Parameter(torch.ones(2))
#         self.weight_yy = nn.Parameter(torch.ones(2))
#         self.weight_xy = nn.Parameter(torch.ones(4))
#
#     def forward(self, displacement):
#         disp_x = displacement[:, 0:1, :, :]  # B x 1 x H x W
#         disp_y = displacement[:, 1:2, :, :]  # B x 1 x H x W
#
#         # 使用固定核计算梯度
#         grad_disp_x_x_cd = F.conv2d(disp_x, self.central_diff_kernel_x, padding=1)
#         grad_disp_x_y_cd = F.conv2d(disp_x, self.central_diff_kernel_y, padding=1)
#         grad_disp_x_x_sobel = F.conv2d(disp_x, self.sobel_kernel_x, padding=1)
#         grad_disp_x_y_sobel = F.conv2d(disp_x, self.sobel_kernel_y, padding=1)
#
#         grad_disp_y_x_cd = F.conv2d(disp_y, self.central_diff_kernel_x, padding=1)
#         grad_disp_y_y_cd = F.conv2d(disp_y, self.central_diff_kernel_y, padding=1)
#         grad_disp_y_x_sobel = F.conv2d(disp_y, self.sobel_kernel_x, padding=1)
#         grad_disp_y_y_sobel = F.conv2d(disp_y, self.sobel_kernel_y, padding=1)
#
#         # 使用可学习的权重组合梯度
#         strain_xx = self.weight_xx[0] * grad_disp_x_x_cd + self.weight_xx[1] * grad_disp_x_x_sobel
#         strain_yy = self.weight_yy[0] * grad_disp_y_y_cd + self.weight_yy[1] * grad_disp_y_y_sobel
#         strain_xy = 0.5 * (
#             self.weight_xy[0] * grad_disp_x_y_cd +
#             self.weight_xy[1] * grad_disp_x_y_sobel +
#             self.weight_xy[2] * grad_disp_y_x_cd +
#             self.weight_xy[3] * grad_disp_y_x_sobel
#         )
#
#         return torch.cat([strain_xx, strain_yy, strain_xy], dim=1)

class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=2, num_classes=3,
                 embed_dim=96, depths=[2, 2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24, 48], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape  # Absolute Position Embedding
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution  # [H_patch, W_patch]
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(
                    depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (
                    i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (
                    self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (
                        self.num_layers - 1 - i_layer)),
                    dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayerUp(
                    dim=int(embed_dim * 2 ** (
                        self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (
                        i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpandX4(
                input_resolution=(
                    img_size // patch_size, img_size // patch_size),
                dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(
                in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        self.strain_layer = StrainLayer()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if (isinstance(m, nn.Linear) and m.bias is not None):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B, L, C

        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - idx -1]], -1)
                x = self.concat_back_dim[idx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B, L, C

        return x

    # def forward_up_features(self, x, x_downsample):
    #     for idx, layer_up in enumerate(self.layers_up):
    #         if idx == 0:
    #             x = layer_up(x)
    #         else:
    #             # Ensure x and x_downsample[...] have the same shape
    #             x_skip = x_downsample[self.num_layers - idx - 1]
    #             # if x.shape != x_skip.shape:
    #             #     # Adjust dimensions if necessary (e.g., using interpolation or linear layers)
    #             #     x_skip = adjust_dimensions(x_skip, x.shape)
    #             x = x * x_skip  # Element-wise multiplication
    #             x = layer_up(x)
    #     x = self.norm_up(x)  # B, L, C
    #     return x

    def upX4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features have wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = self.output(x)

        return x

    def forward(self, img1, img2):
        # Concatenate the two images along the channel dimension
        x = torch.cat([img1, img2], dim=1)  # B, 2, H, W
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.upX4(x)
        x = self.strain_layer(x)
        return x
