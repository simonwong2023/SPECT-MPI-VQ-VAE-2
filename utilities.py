import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
import pandas as pd
import re
import csv

class ImagePairDataset(Dataset):
    def __init__(self, root_dir_label, root_dir_img, transform=None):
        self.root_dir_label = root_dir_label
        self.root_dir_img = root_dir_img
        self.transform = transform
        self.image_list = os.listdir(root_dir_img)  # Load all images in the directory

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        #print(self.root_dir_img)
        #print(self.root_dir_label)

        img_name = os.path.join(self.root_dir_img, self.image_list[idx])
        img = Image.open(img_name)

        img_name_label = os.path.join(self.root_dir_label, self.image_list[idx])
        label = Image.open(img_name_label)

        # Normalize images to [0, 1]

        #print("Image data Max Value: ",np.max(np.array(img)))

        img = np.array(img) / 255
        label = np.array(label) / 255

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return label, img

def save_image_statistics(epoch, save_path, label_images, out_images, phase):
    # Calculate metrics only for the training dataset if phase is 'train'
    if phase == 'train':
        label_images = label_images['train']
        out_images = out_images['train']
        csv_filename = os.path.join(save_path, 'train_image_metrics.csv')
    else:
        label_images = label_images['valid']
        out_images = out_images['valid']
        csv_filename = os.path.join(save_path, 'valid_image_metrics.csv')

    # Calculate total average MAE, MSE, SSIM, and PSNR using FORMULA for (out and label) in each epoch
    avg_mae, avg_mse, avg_ssim, avg_psnr = 0.0, 0.0, 0.0, 0.0
    total_images = len(label_images)
    epsilon = 1e-6  # Small epsilon value to avoid division by zero

    for i in range(total_images):
        label_img = label_images[i]
        out_img = out_images[i]

        # Move tensors to CPU, detach from computation graph, and convert to NumPy arrays
        out_img = np.array(out_img.detach().cpu())
        label_img = np.array(label_img.detach().cpu())

        # Calculate MAE and MSE
        mae = np.mean(np.abs(label_img - out_img))
        mse = np.mean((label_img - out_img) ** 2)

        # Calculate SSIM with dynamic win_size
        min_side = min(label_img.shape)
        win_size = min(7, min_side)  # Set win_size to the smaller side or a maximum of 7

        # Add epsilon to the denominator to avoid division by zero
        ssim = structural_similarity(label_img, out_img, win_size=win_size, data_range=out_img.max() - out_img.min(),
                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(label_img, out_img, data_range=out_img.max() - out_img.min())

        avg_mae += mae
        avg_mse += mse
        avg_ssim += ssim
        avg_psnr += psnr

    # Calculate the average metrics
    avg_mae /= total_images
    avg_mse /= total_images
    avg_ssim /= total_images
    avg_psnr /= total_images

    # Create or append to the CSV file
    if epoch == 0:
        # If it's the first epoch, create the CSV file and write headers
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "MAE", "MSE", "SSIM", "PSNR"])

    # Append the metrics for the current epoch to the CSV file
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, avg_mae, avg_mse, avg_ssim, avg_psnr])

    return avg_mae, avg_mse, avg_ssim, avg_psnr



def metrics_plots(save_path, phase, epoch):

    csv_filename = os.path.join(save_path, f'{phase}_image_metrics.csv')

    if not os.path.exists(csv_filename):
        return

    epoch_values = []
    mae_values = []
    mse_values = []
    ssim_values = []
    psnr_values = []

    with open(csv_filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            epoch_values.append(int(row['Epoch']))
            mae_values.append(float(row['MAE']))
            mse_values.append(float(row['MSE']))
            ssim_values.append(float(row['SSIM']))
            psnr_values.append(float(row['PSNR']))

    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.plot(epoch_values, mae_values, label='MAE', color='blue')  # Adjust color and label
    plt.xlabel('Epoch', fontsize=12)  # Increase fontsize
    plt.ylabel('MAE', fontsize=12)  # Increase fontsize
    plt.xticks(fontsize=10)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=10)  # Increase y-axis tick fontsize
    plt.title(f'{phase} MAE (Epoch {epoch})', fontsize=14)  # Increase title fontsize
    plt.legend(fontsize=10)  # Increase legend fontsize

    plt.subplot(2, 2, 2)
    plt.plot(epoch_values, mse_values, label='MSE', color='green')  # Adjust color and label
    plt.xlabel('Epoch', fontsize=12)  # Increase fontsize
    plt.ylabel('MSE', fontsize=12)  # Increase fontsize
    plt.xticks(fontsize=10)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=10)  # Increase y-axis tick fontsize
    plt.title(f'{phase} MSE (Epoch {epoch})', fontsize=14)  # Increase title fontsize
    plt.legend(fontsize=10)  # Increase legend fontsize

    plt.subplot(2, 2, 3)
    plt.plot(epoch_values, ssim_values, label='SSIM', color='red')  # Adjust color and label
    plt.xlabel('Epoch', fontsize=12)  # Increase fontsize
    plt.ylabel('SSIM', fontsize=12)  # Increase fontsize
    plt.xticks(fontsize=10)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=10)  # Increase y-axis tick fontsize
    plt.title(f'{phase} SSIM (Epoch {epoch})', fontsize=14)  # Increase title fontsize
    plt.legend(fontsize=10)  # Increase legend fontsize

    plt.subplot(2, 2, 4)
    plt.plot(epoch_values, psnr_values, label='PSNR', color='purple')  # Adjust color and label
    plt.xlabel('Epoch', fontsize=12)  # Increase fontsize
    plt.ylabel('PSNR', fontsize=12)  # Increase fontsize
    plt.xticks(fontsize=10)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=10)  # Increase y-axis tick fontsize
    plt.title(f'{phase} PSNR (Epoch {epoch})', fontsize=14)  # Increase title fontsize
    plt.legend(fontsize=10)  # Increase legend fontsize

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{phase}_metrics_epoch_{epoch}.png'))
    plt.close()

def save_loss_plots(n_epochs, latest_epoch, losses, save_path):

    fig = plt.figure(figsize=(15, 15))
    epochs = range(1, latest_epoch + 1)  # Updated to use the latest_epoch
    # BCE Loss
    ax1 = fig.add_subplot(111)

    # Increase font size for axis labels and tick labels
    ax1.tick_params(axis='both', which='major', labelsize=16)  # Adjust the labelsize as needed

    ax1.plot(epochs, losses[0, :latest_epoch, 0], '-')  # Use latest_epoch
    ax1.plot(epochs, losses[1, :latest_epoch, 0], '-')  # Use latest_epoch
    ax1.set_title(f'Total Train and Valid Losses Curves (Epoch {latest_epoch})', fontsize=18)  # Include the epoch number in the title
    ax1.set_xlabel('Epochs', fontsize=16)  # Increase font size for x-axis label
    ax1.set_ylabel('Loss', fontsize=16)  # Increase font size for y-axis label
    ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right", fontsize=16)  # Increase legend font size

    plt.close(fig)
    fig.savefig(f'{save_path}/loss graphs_epoch_{latest_epoch}.png')  # Save with the epoch number


def save_average_train_loss_plots(train_losses, heading_numbers, epoch, save_path):

    plt.figure(figsize=(30, 30))
    for heading_number in heading_numbers:
        # Check if the heading_number exists in train_losses dictionary
        if heading_number in train_losses:
            train_losses_list = [train_losses[heading_number][e] for e in range(1, epoch + 2)]  # Start from 1
            plt.plot(range(1, epoch + 2), train_losses_list, label=f'{heading_number} Train Loss')
    plt.title(f'Average Train Loss Curves (Epoch {epoch})', fontsize=28)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Average Train Loss', fontsize=22)
    plt.xticks(fontsize=18)  # Increase font size of x-axis labels
    plt.yticks(fontsize=18)  # Increase font size of y-axis labels
    plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'average_train_loss_curves_epoch_{epoch}.png'))
    plt.close()

def save_average_valid_loss_plots(valid_losses, heading_numbers, epoch, save_path):

    plt.figure(figsize=(30, 30))
    for heading_number in heading_numbers:
        # Check if the heading_number exists in valid_losses dictionary
        if heading_number in valid_losses:
            valid_losses_list = [valid_losses[heading_number][e] for e in range(1, epoch + 2)]  # Start from 1
            plt.plot(range(1, epoch + 2), valid_losses_list, label=f'{heading_number} Valid Loss')
    plt.title(f'Average Valid Loss Curves (Epoch {epoch})', fontsize=28)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Average Valid Loss', fontsize=22)
    plt.xticks(fontsize=18)  # Increase font size of x-axis labels
    plt.yticks(fontsize=18)  # Increase font size of y-axis labels
    plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'average_valid_loss_curves_epoch_{epoch}.png'))
    plt.close()

