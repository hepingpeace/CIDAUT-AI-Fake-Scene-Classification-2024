import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.cuda.amp as amp
from glob import glob
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoImageProcessor, BeitForImageClassification, AutoModelForImageClassification, SwinForImageClassification, ViTForImageClassification, ConvNextForImageClassification
import shutil
import cv2
from torch_ema import ExponentialMovingAverage
# Log Transform
def log_transform(image):
    image_array = np.array(image, dtype=np.float32)
    image_array[image_array == 0] = 1
    c = 255 / np.log(1 + np.max(image_array))
    log_image = c * np.log(1 + image_array)
    log_image = np.clip(log_image, 0, 255).astype(np.uint8)
    return Image.fromarray(log_image)

# HSV Filter
def hsv_filter(image, hue_shift=0, saturation_scale=1.0, value_scale=1.0):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv_image)
    h = (h + hue_shift) % 180
    s = np.clip(s * saturation_scale, 0, 255)
    v = np.clip(v * value_scale, 0, 255)
    hsv_image = cv2.merge([h, s, v]).astype(np.uint8)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb_image)

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, apply_log=False, apply_hsv=False, transform=None):
        self.annotations = csv_file
        self.img_dir     = img_dir #glob(os.path.join(img_dir, "*.jpg"))
        self.apply_log = apply_log
        self.apply_hsv = apply_hsv
        self.transform   = transform
        assert len(self.img_dir) == len(self.img_dir)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image    = Image.open(img_path).convert("RGB")
        label    = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.apply_log:
            image = log_transform(image)
        if self.apply_hsv:
            image = hsv_filter(image)
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    
def create_dataloaders(csv_file, img_dir, img_size=(224, 224), batch_size=32, n_fold=0):
    # Define transforms with basic augmentations
    # 数据增强
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.GaussianBlur(7),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    # Initialize dataset
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, apply_log=True, apply_hsv=True)
    
    # Create train/validation split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(csv_file)), csv_file.iloc[:, 1].values)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, 
                                 transform=transforms.Compose([transforms.Resize(img_size), 
                                                              transforms.ToTensor()]))
    val_dataset = Subset(dataset, val_index)
    print(len(train_dataset))
    print(len(val_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        
        outputs = model(images).logits[:, :1]
        loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images).logits[:, :1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()  # Convert logits to probabilities
    
    return epoch_loss, all_labels, all_outputs


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def initialize_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

        
def train_model(csv_file, img_dir, model, model_name, img_size=(224, 224), num_epochs=10, batch_size=32, lr=1e-4, n_fold=0, device='cuda', patience=5, warmup_epochs=0):
    train_loader, val_loader = create_dataloaders(csv_file, img_dir, img_size=img_size, batch_size=batch_size, n_fold=n_fold)
    train_sets = len(train_loader)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_sets * warmup_epochs, train_sets * num_epochs)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    train_losses = []
    val_losses = []

    path = model_name + str(n_fold)
    os.makedirs(path, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_losses.append(train_loss)

        # Update EMA after training step
        ema.update()

        # Validate without EMA
        val_loss, val_labels, val_outputs = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Calculate metrics on validation set
        val_preds = (val_outputs > 0.5).astype(int)
        accuracy = accuracy_score(val_labels, val_preds)
        roc_auc = roc_auc_score(val_labels, val_outputs)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val ROC AUC: {roc_auc:.4f}')
        
        # Early stopping check
        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def predict(csv_file, img_dir, model, model_name, img_size=(224, 224), batch_size=32, n_fold=0, device='cuda', delete=False):
    model     = model.to(device) # load the model into the GPU
    model.load_state_dict(torch.load(os.path.join(model_name + str(n_fold), 'checkpoint.pth')))

    # Initialize EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.copy_to()  # Apply EMA parameters to the model
    
    criterion = nn.BCEWithLogitsLoss()
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    test_dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    _, _, outputs = validate(model, test_loader, criterion, device)
    if delete:
        shutil.rmtree(model_name + str(n_fold))
    return outputs


if __name__ == "__main__":
    labels = pd.read_csv ("/home/work/COOOL/faker_image/train.csv")
    labels["label"] = labels["label"].map({"editada":0, "real":1})
    img_dir = "/home/work/COOOL/faker_image/Train"
    model = BeitForImageClassification.from_pretrained('/home/work/COOOL/faker_image/beit-large-patch16-384')
    batch_size = 16
    lr = 1e-6
    img_size = (384, 384)
    n_fold = 0
    train_model(labels, img_dir, model, '/home/work/COOOL/faker_image/beit-large-patch16-384', img_size=img_size, num_epochs=50, 
                batch_size=batch_size, lr=lr, n_fold=n_fold, patience=4, warmup_epochs=0)
    torch.cuda.empty_cache()
    
    labels = pd.read_csv ("//home/work/COOOL/faker_image/sample_submission.csv")
    img_dir = "/home/work/COOOL/faker_image/Test"
    preds = predict(labels, img_dir, model, '/home/work/COOOL/faker_image/beit-large-patch16-384', img_size=img_size, batch_size=batch_size, device='cuda', n_fold=n_fold)
    labels['label'] = preds + 0.005
    labels.to_csv("/home/work/COOOL/faker_image/predictions.csv", index=False)