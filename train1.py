import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import math
import time
import logging
from datetime import datetime
import matplotlib
import numpy as np
import random
import argparse
from sklearn.metrics import recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from utlis.dataloader import ImageDataset

# Import your single-input models here
# Example imports - replace these with your actual model files
from models.resnet import Resnet_152, Resnet_101
from models.densenet import Densenet_121
from models.convnext import ConvNeXt_T, ConvNeXt_S, ConvNeXt_B
from models.swin import Swin_T, Swin_S, Swin_B
from models.vit import ViT

# Use 'Agg' backend for matplotlib to allow image generation without a GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ================= 1. Early Stopping Logic (Identical) =================
class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ================= 2. Evaluation & Logging Helpers (Identical) =================
def setup_logger(output_dir, args):
    log_filename = os.path.join(output_dir, f"training_log.log")
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.info("---------------------------Training Config---------------------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("----------------------------------------------------------------------")
    return logger


def compute_metrics(all_labels, all_preds, num_classes):
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    accuracy, sensitivity, specificity = [], [], []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        accuracy.append((TP + TN) / cm.sum() if cm.sum() != 0 else 0)
        sensitivity.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
        specificity.append(TN / (TN + FP) if (TN + FP) != 0 else 0)
    return accuracy, sensitivity, specificity, cm


def plot_confusion_matrix(cm, epoch, run_dir, writer, num_classes):
    custom_labels = {i: f'Class {i}' for i in range(num_classes)}
    display_labels = [custom_labels[i] for i in range(len(cm))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix (Epoch {epoch})')
    cm_filename = os.path.join(run_dir, f'cm_epoch_{epoch}.png')
    plt.savefig(cm_filename)
    plt.close()
    confusion_image = Image.open(cm_filename)
    writer.add_image(f'Confusion Matrix', transforms.ToTensor()(confusion_image), epoch)


# ================= 3. Core Training & Validation Loop (Modified for Single Input) =================
def train(model, args, device):
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5222, 0.4349, 0.4138], [0.2, 0.1982, 0.1973])
    ])

    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5222, 0.4349, 0.4138], [0.2, 0.1982, 0.1973])
    ])

    train_dataset = ImageDataset(os.path.join(args.dataset_dir, "train"), image_names=args.image_indices,
                                 transform=train_transform)
    val_dataset = ImageDataset(os.path.join(args.dataset_dir, "val"), image_names=args.image_indices,
                               transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return float(ep) / float(max(1, args.warmup_epochs))
        else:
            progress = float(ep - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_dir = os.path.join(args.run_name, args.timestamp, args.fold)
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir, args)
    writer = SummaryWriter(log_dir=run_dir)

    early_stopping = EarlyStopping(patience=args.patience)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train().to(device)
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"{args.fold} Epoch {epoch}/{args.epochs}"):
            # Change: Select only the first image from the tuple for single-input model
            input_img = images[0].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_img)  # Forward pass with single tensor
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                # Change: Select only the first image
                input_img = images[0].to(device)
                labels = labels.to(device)
                outputs = model(input_img)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_accuracy = correct / total
        accuracy, sensitivity, specificity, cm = compute_metrics(all_labels, all_preds, args.num_classes)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        plot_confusion_matrix(cm, epoch, run_dir, writer, args.num_classes)

        log_msg = f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}"
        print(log_msg)
        logger.info(log_msg)

        if val_accuracy >= best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))

        if args.use_early_stop:
            if epoch > args.warmup_epochs and epoch >= args.early_stop_start:
                early_stopping(val_accuracy)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
    writer.close()


# ================= 4. CLI Arguments & Entry Point (Modified Model Selection) =================
def parse_args():
    parser = argparse.ArgumentParser(description="Single-input Model Training Script")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['resnet152', 'resnet101', 'densenet121', 'convnext_t', 'swin_t', 'swin_s', 'swin_b', 'vit'],
                        help="Type of model architecture")
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image_indices', type=int, nargs='+', default=[4], help="Index of single image (default: 4)")
    parser.add_argument('--run_name', type=str, default="Single_Exp")
    parser.add_argument('--timestamp', type=str, required=True)
    parser.add_argument('--fold', type=str, default="fold_1")
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--use_early_stop', action='store_true')
    parser.add_argument('--early_stop_start', type=int, default=30)
    parser.add_argument('--patience', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Define model factory for single-input architectures
    model_factory = {
        'resnet152': Resnet_152,
        'resnet101': Resnet_101,
        'densenet121': Densenet_121,
        'convnext_t': ConvNeXt_T,
        'convnext_s': ConvNeXt_S,
        'convnext_b': ConvNeXt_B,
        'swin_t': Swin_T,
        'swin_s': Swin_S,
        'swin_b': Swin_B,
        'vit': ViT
    }

    if args.model_type not in model_factory:
        raise ValueError(f"Model type {args.model_type} not supported.")

    # Initialize model
    model = model_factory[args.model_type](num_classes=args.num_classes).to(device)

    train(model, args, device)