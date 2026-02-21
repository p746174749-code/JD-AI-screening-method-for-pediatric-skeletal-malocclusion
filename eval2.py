import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.mv_models import ConvNeXt_T3


class TestPairedImageDataset(Dataset):
    """
    Custom Dataset for multi-view inference.
    Loads images 4, 3, and 2 from each sample subfolder.
    """

    def __init__(self, root_dir, image_names, transform=None):
        self.root_dir = root_dir
        self.image_names = image_names
        self.transform = transform
        self.samples = []

        # Sort folders numerically or alphabetically
        subdirs = sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else x)

        for sub in subdirs:
            subdir = os.path.join(root_dir, sub)
            if not os.path.isdir(subdir):
                continue

            paths = []
            for num in image_names:
                fname = f"{num}.jpg"
                found = None
                for f in os.listdir(subdir):
                    if f.lower() == fname.lower():
                        found = os.path.join(subdir, f)
                        break
                if found is None:
                    break
                paths.append(found)

            if len(paths) == len(image_names):
                self.samples.append((sub, paths))

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
        return imgs, int(sample_id)


def main():
    args = parse_args()

    # Setup output and GPU
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Automatically find all best.pth files if a directory is provided
    final_ckpts = []
    for path in args.ckpt_paths:
        if os.path.isdir(path):
            print(f"Searching for 'best.pth' in: {path}")
            for root, dirs, files in os.walk(path):
                if "best.pth" in files:
                    final_ckpts.append(os.path.join(root, "best.pth"))
        elif os.path.isfile(path):
            final_ckpts.append(path)

    final_ckpts = sorted(final_ckpts)  # Ensure fold1, fold2... order
    if not final_ckpts:
        raise FileNotFoundError("No valid checkpoints found!")

    print(f"Found {len(final_ckpts)} checkpoints for ensemble:")
    for c in final_ckpts: print(f" -> {c}")

    # Data transformation and Loader
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5222, 0.4349, 0.4138], [0.2, 0.1982, 0.1973]),
    ])

    test_ds = TestPairedImageDataset(args.test_dir, args.image_names, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load models
    models = []
    for ckpt in final_ckpts:
        model = ConvNeXt_T3(num_classes=args.num_classes).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models.append(model)

    # Inference and Soft Voting
    all_model_results = [[] for _ in range(len(models))]
    avg_results = []

    with torch.no_grad():
        for imgs_list, sample_id in test_loader:
            inputs = [im.to(device, non_blocking=True) for im in imgs_list]
            probs_models = []

            for i, model in enumerate(models):
                logits = model(*inputs)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                # Store individual model row
                row = {"id": sample_id.item(), "pred": probs.argmax() + 1}
                for c in range(args.num_classes):
                    row[f"prob_class{c + 1}"] = probs[c]
                all_model_results[i].append(row)
                probs_models.append(probs)

            # Calculate Ensemble Average
            avg_probs = sum(probs_models) / len(probs_models)
            row_avg = {"id": sample_id.item(), "pred": avg_probs.argmax() + 1}
            for c in range(args.num_classes):
                row_avg[f"prob_class{c + 1}"] = avg_probs[c]
            avg_results.append(row_avg)

    # Export results
    for i, results in enumerate(all_model_results, start=1):
        pd.DataFrame(results).sort_values("id").to_csv(os.path.join(args.out_dir, f"model{i}.csv"), index=False)

    pd.DataFrame(avg_results).sort_values("id").to_csv(os.path.join(args.out_dir, "avg.csv"), index=False)
    print(f"Results saved to {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--ckpt_paths", type=str, nargs="+", required=True,
                        help="Can be a directory or list of .pth files")
    parser.add_argument("--image_names", type=int, nargs="+", default=[4, 3, 2])
    parser.add_argument("--out_dir", type=str, default="results-figure")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()