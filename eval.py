import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# Assuming your models are stored in models/mv_models.py
from models.mv_models import ConvNeXt_T2, ConvNeXt_T3, ConvNeXt_T4


class TestPairedImageDataset(Dataset):
    """
    Dataset for testing: Each subfolder under root_dir represents one sample.
    Loads JPG images corresponding to indices specified in IMAGE_NAMES.
    __getitem__ returns (list_of_images, sample_id)
    """

    def __init__(self, root_dir, image_names, transform=None):
        self.root_dir = root_dir
        self.image_names = image_names
        self.transform = transform
        self.samples = []  # list of (sample_id, [paths...])

        # Sort folders numerically if possible, otherwise alphabetically
        subdirs = sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else x)

        for sub in subdirs:
            subdir = os.path.join(root_dir, sub)
            if not os.path.isdir(subdir):
                continue

            # Locate all specified image indices
            paths = []
            for num in image_names:
                fname = f"{num}.jpg"
                # Case-insensitive search
                found = None
                for f in os.listdir(subdir):
                    if f.lower() == fname.lower():
                        found = os.path.join(subdir, f)
                        break
                if found is None:
                    break
                paths.append(found)

            # Add sample only if all required images are found
            if len(paths) == len(image_names):
                self.samples.append((sub, paths))
            else:
                print(f"[WARN] Not all images found in {subdir}, skipping...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, paths = self.samples[idx]
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # Returns a list of tensors and the sample ID
        return imgs, int(sample_id)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # Normalization values based on your specific training dataset stats
        transforms.Normalize([0.5222, 0.4349, 0.4138], [0.2, 0.1982, 0.1973]),
    ])

    # Build Test Dataset
    test_ds = TestPairedImageDataset(
        root_dir=args.test_dir,
        image_names=args.image_names,
        transform=transform
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model Factory for Hyperparameter Control
    model_factory = {
        "convnext_t2": ConvNeXt_T2,
        "convnext_t3": ConvNeXt_T3,
        "convnext_t4": ConvNeXt_T4
    }

    if args.model_type not in model_factory:
        raise ValueError(f"Model type {args.model_type} not supported. Choose from {list(model_factory.keys())}")

    # Initialize model based on hyperparameter
    print(f"Loading model architecture: {args.model_type}")
    model = model_factory[args.model_type](num_classes=args.num_classes).to(device)

    # Load Checkpoint
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []
    print("Starting Inference...")
    with torch.no_grad():
        for imgs_list, sample_id in test_loader:
            # Handle multi-input vs single-input branching
            if isinstance(imgs_list, (list, tuple)):
                inputs = [im.to(device, non_blocking=True) for im in imgs_list]
                logits = model(*inputs)
            else:
                logits = model(imgs_list.to(device, non_blocking=True))

            # Get prediction (assuming class labels start from 1 in your target CSV)
            preds = logits.argmax(dim=1).cpu().numpy()
            sample_ids = sample_id.numpy()

            for i in range(len(preds)):
                results.append({"id": int(sample_ids[i]), "pred": int(preds[i]) + 1})

    # Save to CSV
    df = pd.DataFrame(results).sort_values("id")
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-View Inference Script")
    # Dataset Parameters
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test set root directory")
    parser.add_argument("--image_names", type=int, nargs="+", default=[4, 3, 2],
                        help="Indices of images to load (e.g. 4 3 2)")

    # Model Parameters
    parser.add_argument("--model_type", type=str, default="convnext_t3",
                        choices=["convnext_t2", "convnext_t3", "convnext_t4"],
                        help="Which multi-view architecture to use")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of output classes")

    # Execution Parameters
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--out_csv", type=str, default="test_preds.csv", help="Output filename for predictions")

    return parser.parse_args()


if __name__ == "__main__":
    main()