import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from models.mv_models import ConvNeXt_T3

# ============ Image Processing Functions ============

resize_input = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()


def load_image(path):
    """Load image, resize for model input, and return both tensor and original image."""
    img = Image.open(path).convert('RGB')
    resized_img = resize_input(img)
    tensor = to_tensor(resized_img).unsqueeze(0)
    return tensor, img


def overlay_cam_on_image(pil_img, cam_tensor):
    """Overlay the heatmap on the original image using OpenCV COLORMAP_JET."""
    img = np.array(pil_img)
    if img.dtype != np.uint8:
        img = np.uint8(255 * img / (img.max() + 1e-8))

    heatmap = cam_tensor.squeeze().cpu().detach().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    superimposed = heatmap * 0.5 + img * 0.5
    return np.uint8(superimposed)


def generate_gradcam(feature_map, gradient, output_size):
    """Generate Grad-CAM heatmap from feature maps and gradients."""
    weights = gradient.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feature_map).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=output_size, mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam


# ============ Grad-CAM Calculation for Single Image ============

def calc_cam_for_single_img(model, input_img, orig_img_size, device):
    """Calculate Grad-CAM for a single input using hooks on the specific backbone layer."""
    feature_map = {}
    gradients = {}

    def save_feature_hook(module, input, output):
        feature_map['feat'] = output.detach()

    def save_gradient_hook(module, grad_input, grad_output):
        gradients['grad'] = grad_output[0].detach()

    # Register hooks on the last feature layer of the backbone
    handle_f = model.shared_backbone.features[7].register_forward_hook(save_feature_hook)
    handle_b = model.shared_backbone.features[7].register_full_backward_hook(save_gradient_hook)

    model.zero_grad()
    # Note: Model expects 3 inputs for this specific architecture
    output = model(input_img, input_img, input_img)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    score.backward()

    # Clean up hooks
    handle_f.remove()
    handle_b.remove()

    cam = generate_gradcam(feature_map['feat'], gradients['grad'], output_size=orig_img_size[::-1])
    return cam


# ============ Execution Logic ============

def run_visualization(model_path, test_dir, output_root, num_classes, device):
    """Iterate through folders and save Grad-CAM visualizations."""
    model = ConvNeXt_T3(num_classes=num_classes, share_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist.")
        return

    for folder_name in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        img_paths = {
            'x1': os.path.join(folder_path, '4.jpg'),
            'x2': os.path.join(folder_path, '3.jpg'),
            'x3': os.path.join(folder_path, '2.jpg'),
        }

        if not all(os.path.exists(p) for p in img_paths.values()):
            print(f"[SKIP] {folder_name} is missing required images.")
            continue

        print(f"Processing: {folder_name}")
        save_dir = os.path.join(output_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Load images
        inputs = {}
        for key in ['x1', 'x2', 'x3']:
            tensor, orig = load_image(img_paths[key])
            inputs[key] = (tensor.to(device), orig)

        # Calculate and save Grad-CAM for each view
        for name in ['x1', 'x2', 'x3']:
            tensor, orig = inputs[name]
            cam = calc_cam_for_single_img(model, tensor, orig.size, device)
            vis_img = overlay_cam_on_image(orig, cam)

            plt.imshow(vis_img)
            plt.axis('off')
            plt.title(f"Grad-CAM: {name}")
            plt.savefig(os.path.join(save_dir, f'gradcam_{name}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f"All results saved to: {output_root}")


if __name__ == '__main__':
    # ====== SETTINGS & PATHS ======
    MODEL_WEIGHTS = 'checkpoints/runs-convnext-432/fold2/best.pth'
    DATA_DIR = 'data/data-test'
    SAVE_ROOT = 'results_gradcam'
    CLASSES = 3

    # System Setup
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {current_device}")

    # Start Processing
    run_visualization(
        model_path=MODEL_WEIGHTS,
        test_dir=DATA_DIR,
        output_root=SAVE_ROOT,
        num_classes=CLASSES,
        device=current_device
    )