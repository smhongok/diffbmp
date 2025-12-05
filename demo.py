import torch    
import pydiffbmp    
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import matplotlib.pyplot as plt

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 0. Load target image (optional - create synthetic target if no image)
target_size = (256, 256)
transform_pipeline = transforms.Compose([
    transforms.Resize(target_size), 
    transforms.ToTensor(),         
    transforms.Lambda(lambda t: t.permute(1, 2, 0))
])
img_pil = Image.open("images/person/vangogh1.jpg").convert('RGB')
target = transform_pipeline(img_pil).to(device)
print(f"Tensor shape (C, H, W): {target.shape}")

# 1. Primitive 로딩
primitive = pydiffbmp.load_primitive("assets/primitives/flowers/Gerbera1.png", size=128, device=device)

# 2. 파라미터 초기화 (requires_grad=True로 자동 설정)
x, y, r, theta, v, c = pydiffbmp.initialize_params(
    n_primitives=1000,
    canvas_size=target_size,
    method='structure_aware',  # or 'structure_aware' with target_image=target
    target_image=target,
    device=device
)

# 3. Setup optimizer
optimizer = torch.optim.Adam([x, y, r, theta, v, c], lr=0.1)

# 4. Optimization loop - 사용자가 완전히 제어
print("Starting optimization...")
for step in range(100):
    # Differentiable render
    rendered = pydiffbmp.render(
        primitive, x, y, r, theta, v, c, 
        canvas_size=target_size,
        background='white'
    )  # returns torch (H, W, 3)
    
    # 사용자 정의 loss
    loss = F.mse_loss(rendered[:, :, :3], target)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log progress
    if step % 10 == 0:
        print(f"Step {step:4d}, Loss: {loss.item():.6f}")

print("Optimization complete!")

# 5. Save result
with torch.no_grad():
    final_rendered = pydiffbmp.render(
        primitive, x, y, r, theta, v, c,
        canvas_size=target_size,
        background='white'
    )
    
    # Save as image
    output_np = (final_rendered[:, :, :3].cpu().numpy() * 255).astype('uint8')
    Image.fromarray(output_np).save('demo_output.png')
    print("Saved result to demo_output.png")
    
    # Also save target for comparison
    target_np = (target.cpu().numpy() * 255).astype('uint8')
    Image.fromarray(target_np).save('demo_target.png')
    print("Saved target to demo_target.png")