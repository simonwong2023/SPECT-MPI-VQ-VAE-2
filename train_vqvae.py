import argparse
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from networks import VQVAE
from utilities import ImagePairDataset, save_loss_plots, save_average_train_loss_plots, save_average_valid_loss_plots, save_image_statistics, metrics_plots
from torchvision import transforms
from torchvision.utils import save_image
import re
import cv2
import csv

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=2000)  # Increase the number of epochs for continuing training
parser.add_argument('--lr', type=float, default=3e-7)
parser.add_argument('--first_stride', type=int, default=4, help="2, 4, 8, or 16")
parser.add_argument('--second_stride', type=int, default=2, help="2, 4, 8, or 16")
parser.add_argument('--embed_dim', type=int, default=64)
#parser.add_argument('--data_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets_Grayscale/HT REST')
#parser.add_argument('--dataset', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets_Grayscale/FT REST')
#parser.add_argument('--data_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST HT (Testing 256x256x16)')
#parser.add_argument('--dataset', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST FT_HT (Testing 256x256x16)')
parser.add_argument('--data_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST HT (Training 256x256x16)')
parser.add_argument('--dataset', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST FT_HT (Training 256x256x16)')
parser.add_argument('--view', type=str, default='frontal', help="frontal or lateral")
parser.add_argument('--save_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/save')
parser.add_argument('--train_run', type=str, default='0')
parser.add_argument('--model_checkpoint', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/checkpoint/vqvae_010.pt')  # Add this argument for model checkpoint
#parser.add_argument('--model_checkpoint', type=str, default='None')  # Add this argument for model checkpoint
#parser.add_argument('--csv_file', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets_Grayscale/CSV/out_image_values.csv')
args = parser.parse_args()
torch.manual_seed(816)

save_path = f'{args.save_path}/{args.dataset}/{args.train_run}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/checkpoint/', exist_ok=True)
os.makedirs(f'{save_path}/sample/', exist_ok=True)
with open(f'{save_path}/args.txt', 'w') as f:
    for key in vars(args).keys():
        f.write(f'{key}: {vars(args)[key]}\n')
        print(f'{key}: {vars(args)[key]}')


# Modify your data loader creation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    # Add any other transformations you need here
])

# Create the dataset instances
train_dataset = ImagePairDataset(root_dir_label=args.dataset, root_dir_img=args.data_path, transform=transform)
valid_dataset = ImagePairDataset(root_dir_label=args.dataset, root_dir_img=args.data_path, transform=transform)

# Define batch size (change as needed)
batch_size = 2
# Define the number of workers for data loading
num_workers = 4  # You can adjust this number based on your system's capabilities

# Create data loaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

# Example usage of the dataloader
for i, img in enumerate(dataloaders['train']):
    sample_img = [Variable(item.type(Tensor)) for item in img]


# Initialize the VQ-VAE model
if os.path.exists(args.model_checkpoint):
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.model_checkpoint))  # Load the pre-trained model checkpoint
    print("Loaded pre-trained model from checkpoint:", args.model_checkpoint)
else:
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)


if cuda:
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim).cuda()
else:
    model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    device_ids = list(range(n_gpu))
    model = nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

losses = np.zeros((2, args.n_epochs, 3))  # [0,:,:] index for train, [1,:,:] index for valid


def get_heading_number(image_filename):
    # Extract the heading_number from the image filename
    heading_match = re.search(r'(\d+T)\s', image_filename)
    if heading_match:
        heading_number = heading_match.group(1)
        return heading_number
    else:
        return None
def get_batch_number(image_filename):
    # Extract the batch_number from the image filename
    batch_match = re.search(r'(\d+)\s', image_filename)
    if batch_match:
        batch_number = batch_match.group(1)
        return batch_number
    else:
        return None

# Initialize a dictionary to store average losses for different heading numbers
average_losses = {}
train_losses = {}  # Separate dictionaries for train and valid losses
valid_losses = {}
heading_numbers = set()  # Keep track of unique heading numbers

# Create folders to save plots
os.makedirs(os.path.join(save_path, 'loss_graphs'), exist_ok=True)
#os.makedirs(os.path.join(save_path, 'train_metrics'), exist_ok=True)
#os.makedirs(os.path.join(save_path, 'valid_metrics'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'average_train_loss_curves'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'average_valid_loss_curves'), exist_ok=True)

# Within the training loop
for epoch in range(args.n_epochs):
    for phase in ['train', 'valid']:
        model.train(phase == 'train')  # True when 'train', False when 'valid'
        criterion = nn.MSELoss()

        latent_loss_weight = 0.25
        n_row = 1  # Display one reconstructed image per figure
        loader = tqdm(dataloaders[phase])

        # Initialize a dictionary to store losses for different heading numbers
        heading_losses = {}

        # Lists to store label and out images for statistics calculation
        label_images = {'train': [], 'valid': []}  # Store label images for metrics calculation
        out_images = {'train': [], 'valid': []}  # Store out images for metrics calculation

        for i, (img, label) in enumerate(loader):
            img = Variable(img.type(Tensor))
            with torch.set_grad_enabled(phase == 'train'):
                optimizer.zero_grad()

                # Print the maximum and minimum pixel values of the img image tensor
                #print("Max pixel value in HT image:", torch.max(img))
                #print("Min pixel value in HT image:", torch.min(img))
                # Print the shape of the input image tensor
                #print("Input HT image shape before feeding model:", img.shape)
                #print("FT image shape:", label.shape)
                #print(f"Batch {i + 1}/{len(loader)}")

                # Extract and print the filenames of the images
                img_filenames = train_dataset.image_list[i * batch_size:(i + 1) * batch_size]
                label_filenames = [os.path.join(args.dataset, fname) for fname in img_filenames]
                #print("Image Filenames:")
                #for img_filename, label_filename in zip(img_filenames, label_filenames):
                    #print(f"Image: {img_filename}, Label: {label_filename}")

                # Extract heading_number
                heading_number = get_heading_number(img_filenames[0])
                if heading_number is not None:
                    #print(f"Heading Number: {heading_number}")
                    heading_numbers.add(heading_number)

                out, latent_loss = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()
                loss = recon_loss + latent_loss_weight * latent_loss

                # Print the MSE loss value for this batch
                #print(f'Batch {i + 1}/{len(loader)} - {phase} MSE loss: {recon_loss.item():.5f}')

                # Print values of the "out" image tensor
                # print("Values of 'out' image:")
                # print(out)

                # Save the "out" images in folders based on heading_number and epoch
                for j in range(len(img_filenames)):
                    img_filename = img_filenames[j]
                    heading_number = get_heading_number(img_filename)
                    if heading_number is not None:
                        # Create a folder for each heading_number and epoch if it doesn't exist
                        save_folder = os.path.join(save_path, 'sample', f'{heading_number} REST_epoch {epoch + 1}')
                        os.makedirs(save_folder, exist_ok=True)
                        out_filename = os.path.join(save_folder,
                                                    f'{os.path.basename(img_filename)}_epoch_{epoch + 1}.png')
                        save_image(out[j].data, out_filename, normalize=False)

                        # Print the names of the saved "out" images
                        #print(f"Saved 'out' image: {out_filename}")

                        # Store the loss for this heading number
                        if heading_number in heading_losses:
                            heading_losses[heading_number].append(loss.item())
                        else:
                            heading_losses[heading_number] = [loss.item()]

                        # Separate train and valid losses
                        if phase == 'train':
                            if heading_number in train_losses:
                                train_losses[heading_number].append(loss.item())
                            else:
                                train_losses[heading_number] = [loss.item()]
                        else:
                            if heading_number in valid_losses:
                                valid_losses[heading_number].append(loss.item())
                            else:
                                valid_losses[heading_number] = [loss.item()]

                        # Store label and out images for metrics calculation
                        if phase == 'train':
                            label_images['train'].extend(label)
                            out_images['train'].extend(out)
                        else:
                            label_images['valid'].extend(label)
                            out_images['valid'].extend(out)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # Convert CUDA tensors to NumPy arrays after moving them to CPU
                    losses[0, epoch, :] = [loss.cpu().item(), recon_loss.cpu().item(), latent_loss.cpu().item()]
                else:
                    # Convert CUDA tensors to NumPy arrays after moving them to CPU
                    losses[1, epoch, :] = [loss.cpu().item(), recon_loss.cpu().item(), latent_loss.cpu().item()]
                lr = optimizer.param_groups[0]['lr']

            loader.set_description((f'phase: {phase}; epoch: {epoch + 1}; total_loss: {loss.item():.5f}; '
                                    f'latent: {latent_loss.item():.5f}; mse: {recon_loss.item():.5f}; '
                                    f'lr: {lr:.8f}'))

        # Calculate and display average losses for different heading numbers
        for heading_number, losses_list in heading_losses.items():
            average_loss = np.mean(losses_list)
            #print(f'Average {phase} loss for heading_number {heading_number}: {average_loss:.5f}')

            # Store average losses for this heading number and epoch
            if heading_number in average_losses:
                average_losses[heading_number][epoch] = average_loss
            else:
                average_losses[heading_number] = {epoch: average_loss}

        # Save image statistics
        save_image_statistics(epoch, save_path, label_images, out_images, phase)

        # Save loss plots and metrics plots at the end of each epoch
        save_loss_plots(args.n_epochs, epoch + 1, losses, f'{save_path}/loss_graphs')
        #metrics_plots(f'{save_path}/train_metrics', 'train', epoch + 1)
        #metrics_plots(f'{save_path}/valid_metrics', 'valid', epoch + 1)
        metrics_plots(f'{save_path}', 'train', epoch+ 1)
        metrics_plots(f'{save_path}', 'valid', epoch+ 1)
        save_average_train_loss_plots(train_losses, heading_numbers, epoch + 1, f'{save_path}/average_train_loss_curves')
        save_average_valid_loss_plots(valid_losses, heading_numbers, epoch + 1, f'{save_path}/average_valid_loss_curves')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{save_path}/checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt')
