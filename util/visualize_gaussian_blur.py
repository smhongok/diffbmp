import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svg_loader import SVGLoader
from utils import gaussian_blur
import matplotlib.colors as mcolors


### !!!!! 이 함수는 main 에서 복붙 하여서 사용 !!!!! ###
def batched_soft_rasterize(bmp_image, X, Y, x, y, r, theta, sigma=0.0):
    """
    bmp_image: (H_bmp, W_bmp) tensor in [0,1]
    X, Y: (1, H, W) coordinate grids
    x, y, r, theta: each is tensor of shape (B,)
    sigma: Gaussian blur standard deviation for softening the mask
    Returns:
        (B, H, W) soft masks
    """
    B = len(x)
    _, H, W = X.shape

    # 1) 입력 bmp_image에 Gaussian blur 적용 (soft rasterization 효과)
    if sigma > 0.0:
        # bmp_image: (H_bmp, W_bmp) -> (1,H_bmp,W_bmp)
        bmp = bmp_image.unsqueeze(0)
        # gaussian_blur: (N, H, W) 형태를 기대하므로 N=1
        bmp = gaussian_blur(bmp, sigma)  # (N, H_bmp, W_bmp)
        bmp_image = bmp.squeeze(0)       # (H_bmp, W_bmp)

    # 2) 그리드와 파라미터를 배치 크기에 맞춰 확장
    X_exp = X.expand(B, H, W)
    Y_exp = Y.expand(B, H, W)
    x_exp = x.view(B,1,1).expand(B, H, W)
    y_exp = y.view(B,1,1).expand(B, H, W)
    r_exp = r.view(B,1,1).expand(B, H, W)

    # 3) 위치 정규화 및 회전
    pos = torch.stack([X_exp - x_exp, Y_exp - y_exp], dim=1) / r_exp.unsqueeze(1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R_inv = torch.zeros(B, 2, 2, device=X.device)
    R_inv[:,0,0] = cos_t; R_inv[:,0,1] = sin_t
    R_inv[:,1,0] = -sin_t; R_inv[:,1,1] = cos_t
    uv = torch.einsum('bij,bjhw->bihw', R_inv, pos)

    # 4) grid_sample을 위한 형태 조정
    grid = uv.permute(0,2,3,1)  # (B, H, W, 2)
    bmp_exp = bmp_image.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,1,H_bmp,W_bmp)

    # 5) bilinear sampling
    sampled = F.grid_sample(bmp_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return sampled.squeeze(1)  # (B, H, W)

def visualize_gaussia_blur():
    # --- 1) SVG 알파 마스크 로드 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    svg_path = 'images/apple_logo_hollow.svg'  # 시각화할 SVG 파일 경로
    loader = SVGLoader(svg_path, output_width=128, device=device)
    bmp_tensor = loader.load_alpha_bitmap()
    bmp_np = bmp_tensor.cpu().numpy()    # (H_bmp, W_bmp)
    size = bmp_np.shape[0]

    # --- 2) 정규화된 좌표 그리드 생성 [-1,1] ---
    xs = np.linspace(-1,1,size)
    ys = np.linspace(-1,1,size)
    Xg, Yg = np.meshgrid(xs, ys)
    x_lin = torch.linspace(-1,1,size,device=device)
    y_lin = torch.linspace(-1,1,size,device=device)
    Xc, Yc = torch.meshgrid(x_lin, y_lin, indexing='xy')
    Xc, Yc = Xc.unsqueeze(0), Yc.unsqueeze(0)

    # --- 3) 파라미터 설정: 중심(x,y)=0, 반지름=1, 회전=0 ---
    B = 1
    x = torch.zeros(B, device=device)
    y = torch.zeros(B, device=device)
    r = torch.ones(B, device=device)
    theta = torch.zeros(B, device=device)

    # --- 4) sigma 값에 따른 마스크 생성 ---
    sigmas = [0.0, 1.0, 2.0]
    masks = []
    for s in sigmas:
        out = batched_soft_rasterize(bmp_tensor, Xc, Yc, x, y, r, theta, sigma=s)
        masks.append(out[0].cpu().numpy())
        
    # --- 5) 2D heatmap 시각화 설정 ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom', [(1,1,1), (1,0,0), (0,0,0)])

    # 그림 크기를 조정하고 오른쪽에 컬러바 공간 확보
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    # 여백 조정: 서브플롯을 왼쪽으로 이동, 오른쪽에 컬러바 공간 확보
    fig.subplots_adjust(left=0.05, right=0.80, wspace=0.15, top=0.85, bottom=0.15)
    im_list = []
    # 각 서브플롯에 heatmap 그리기
    for ax, mask, s in zip(axes, masks, sigmas):
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        # sigma를 그리스문자로: '\sigma'
        ax.set_title('$\sigma='+f'{s}$', fontsize=16)
        ax.axis('off')
        im_list.append(im)
    # 오른쪽에 세로 컬러바 추가 (서브플롯과 겹치지 않도록)
    cbar = fig.colorbar(im_list[0], ax=axes, orientation='vertical', fraction=0.05, pad=0.05, aspect=20)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Mask Value', fontsize=14)
    plt.savefig('svg_soft_raster_profiles_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_gaussia_blur()