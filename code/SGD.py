# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix
import logging
import sys
from datetime import datetime

# Setup logging - specify logger directory
def setup_logging(optimizer_name='SGD'):
    # Specify log file directory
    logs_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/logger'
    
    # Ensure logger directory exists
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"Created logger directory: {logs_dir}")
    except Exception as e:
        print(f"Error creating logger directory: {e}")
        # If creation fails, use current directory
        logs_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Using fallback directory: {logs_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'training_{optimizer_name}_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add a test log record
    logging.info(f"Logging initialized for {optimizer_name} optimizer")
    logging.info(f"Log file: {log_filename}")
    print(f"Log file created at: {log_filename}")
    
    return log_filename

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256), is_test=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.is_test = is_test
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(image_extensions)])
        
        self.mask_files = []
        valid_image_files = []
        
        for img_file in self.image_files:
            mask_candidates = [
                os.path.splitext(img_file)[0] + '.png',
                os.path.splitext(img_file)[0] + '.jpg',
                img_file
            ]
            
            mask_found = False
            for mask_candidate in mask_candidates:
                mask_path = os.path.join(mask_dir, mask_candidate)
                if os.path.exists(mask_path):
                    self.mask_files.append(mask_candidate)
                    valid_image_files.append(img_file)
                    mask_found = True
                    break
            
            if not mask_found and not is_test:
                logging.warning(f"No mask file found for {img_file}")
        
        self.image_files = valid_image_files
        
        logging.info(f"Found {len(self.image_files)} valid image-mask pairs")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path)
        
        original_size = image.size
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        image = np.array(image)
        mask = np.array(mask)
        
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        
        mask = np.clip(mask, 0, 5)
        
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask, self.image_files[idx], original_size

# Simple U-Net Implementation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(UNet, self).__init__()
        
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self._block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.conv_out(d1)

# Evaluation Metrics Calculation
def calculate_metrics(pred, target, num_classes=6):
    pred = torch.argmax(pred, dim=1)
    
    ious = []
    pixel_accuracy = []
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
        
        if target_inds.sum() > 0:
            class_acc = (pred_inds & target_inds).sum().float() / target_inds.sum().float()
            pixel_accuracy.append(class_acc.item())
        else:
            pixel_accuracy.append(float('nan'))
    
    overall_acc = (pred == target).sum().float() / target.numel()
    
    return np.nanmean(ious), overall_acc.item(), np.nanmean(pixel_accuracy)

# Training Function
def train_model(model, train_loader, val_loader, optimizer_name='SGD', num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Training with {optimizer_name} optimizer")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD optimizer with momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_ious = []
    val_accuracies = []
    best_val_iou = 0.0
    
    logging.info(f"Starting training with {optimizer_name} optimizer for {num_epochs} epochs")
    logging.info(f"Learning rate: {lr}, Momentum: 0.9, Weight decay: 1e-4")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{optimizer_name}-Train]')
        
        for images, masks, _, _ in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{optimizer_name}-Val]')
            for images, masks, _, _ in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                iou, accuracy, _ = calculate_metrics(outputs, masks)
                total_iou += iou
                total_accuracy += accuracy
                
                val_bar.set_postfix(loss=loss.item(), iou=iou, acc=accuracy)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        avg_val_accuracy = total_accuracy / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_accuracies.append(avg_val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            checkpoint_path = f'/mnt/nvme1/suyuejiao/egohos_split_data/ckpt_{optimizer_name}/best_model.pth'
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_iou': best_val_iou,
                'val_loss': avg_val_loss,
                'optimizer_name': optimizer_name
            }, checkpoint_path)
            logging.info(f"[{optimizer_name}] Saved best model: {checkpoint_path}")
            logging.info(f"[{optimizer_name}] Best Val IoU: {avg_val_iou:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        logging.info(f'[{optimizer_name}] Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Acc: {avg_val_accuracy:.4f}')
    
    logging.info(f"[{optimizer_name}] Training completed. Best validation IoU: {best_val_iou:.4f}")
    return train_losses, val_losses, val_ious, val_accuracies

# Testing Function
def test_model(model, test_loader, optimizer_name='SGD'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    total_class_accuracy = 0.0
    
    all_predictions = []
    all_targets = []
    
    logging.info(f"[{optimizer_name}] Starting testing...")
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f'Testing [{optimizer_name}]')
        for images, masks, filenames, original_sizes in test_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            iou, accuracy, class_accuracy = calculate_metrics(outputs, masks)
            total_iou += iou
            total_accuracy += accuracy
            total_class_accuracy += class_accuracy
            
            pred = torch.argmax(outputs, dim=1)
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            test_bar.set_postfix(loss=loss.item(), iou=iou, acc=accuracy)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_iou = total_iou / len(test_loader)
    avg_test_accuracy = total_accuracy / len(test_loader)
    avg_class_accuracy = total_class_accuracy / len(test_loader)
    
    cm = confusion_matrix(all_targets, all_predictions, labels=range(6))
    
    logging.info(f"\n[{optimizer_name}] Test Results:")
    logging.info(f"[{optimizer_name}] Test Loss: {avg_test_loss:.4f}")
    logging.info(f"[{optimizer_name}] Test IoU: {avg_test_iou:.4f}")
    logging.info(f"[{optimizer_name}] Test Pixel Accuracy: {avg_test_accuracy:.4f}")
    logging.info(f"[{optimizer_name}] Test Class Average Accuracy: {avg_class_accuracy:.4f}")
    
    return {
        'test_loss': avg_test_loss,
        'test_iou': avg_test_iou,
        'test_accuracy': avg_test_accuracy,
        'test_class_accuracy': avg_class_accuracy,
        'confusion_matrix': cm
    }

# Main Function
def main():
    optimizer_name = 'SGD'
    
    # First setup logging with optimizer name
    log_filename = setup_logging(optimizer_name)
    
    # Ensure logging is working
    logging.info(f"Starting training process with {optimizer_name} optimizer")
    print(f"Log file location: {log_filename}")
    
    # Check if log file is actually created
    if os.path.exists(log_filename):
        logging.info(f"Log file successfully created at specified location: {log_filename}")
    else:
        print(f"WARNING: Log file was not created at expected location: {log_filename}")
        # Try to create in current directory
        current_log = f"/mnt/nvme1/suyuejiao/egohos_split_data/logger/training_{optimizer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(current_log, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"Using fallback log file: {current_log}")
    
    train_image_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/train/image'
    train_mask_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/train/lbl_hand_1stobj'
    val_image_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/test_indomain/image'
    val_mask_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/test_indomain/lbl_hand_obj1st'
    
    # Create output directories if they don't exist
    image_output_dir = '/mnt/nvme1/suyuejiao/egohos_split_data/image'
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
        logging.info(f"Created image output directory: {image_output_dir}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=transform,
        image_size=(256, 256)
    )
    
    val_dataset = SegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=transform,
        image_size=(256, 256),
        is_test=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    
    # Use SGD optimizer
    logging.info(f"\n{'='*60}")
    logging.info(f"Training with {optimizer_name} optimizer")
    logging.info(f"{'='*60}")
    
    model = UNet(in_channels=3, out_channels=6)
    
    train_losses, val_losses, val_ious, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer_name,
        num_epochs=50, lr=0.001
    )
    
    results = {
        optimizer_name: {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_ious': val_ious,
            'val_accuracies': val_accuracies
        }
    }
    
    # Plot training curves
    plt.figure(figsize=(20, 12))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(results[optimizer_name]['train_losses'], label=optimizer_name, linewidth=2.5, color='#1f77b4')
    plt.title(f'Training Loss ({optimizer_name} Optimizer)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(results[optimizer_name]['val_losses'], label=optimizer_name, linewidth=2.5, color='#1f77b4')
    plt.title(f'Validation Loss ({optimizer_name} Optimizer)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # IoU
    plt.subplot(2, 2, 3)
    plt.plot(results[optimizer_name]['val_ious'], label=optimizer_name, linewidth=2.5, color='#1f77b4')
    plt.title(f'Validation IoU ({optimizer_name} Optimizer)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(results[optimizer_name]['val_accuracies'], label=optimizer_name, linewidth=2.5, color='#1f77b4')
    plt.title(f'Validation Accuracy ({optimizer_name} Optimizer)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plot_image_path = f'/mnt/nvme1/suyuejiao/egohos_split_data/image/{optimizer_name.lower()}_training_curves.png'
    plt.savefig(plot_image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logging.info(f"Saved training curves: {plot_image_path}")
    
    # Save results to JSON file
    results_path = f'/mnt/nvme1/suyuejiao/egohos_split_data/image/{optimizer_name.lower()}_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved training results to: {results_path}")
    
    # Print final results
    final_iou = results[optimizer_name]['val_ious'][-1]
    final_acc = results[optimizer_name]['val_accuracies'][-1]
    final_loss = results[optimizer_name]['val_losses'][-1]
    
    logging.info("\n" + "="*80)
    logging.info(f"Final Training Results ({optimizer_name} Optimizer):")
    logging.info("="*80)
    logging.info(f"Final Validation IoU: {final_iou:.4f}")
    logging.info(f"Final Validation Accuracy: {final_acc:.4f}")
    logging.info(f"Final Validation Loss: {final_loss:.4f}")
    logging.info("="*80)
    
    logging.info(f"\nTraining with {optimizer_name} optimizer completed! All results saved.")

if __name__ == '__main__':
    main()