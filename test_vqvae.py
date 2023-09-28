import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import save_image
from networks import VQVAE
from utilities import ImagePairDataset, transforms

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST HT (Training 256x256x16)')
parser.add_argument('--save_path', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/train REST')
parser.add_argument('--model_checkpoint', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/checkpoint/vqvae_010.pt')
parser.add_argument('--dataset', type=str, default='/home/simon/image_reconstruction-master/vqvae2-master/Datasets Grayscale/Removed REST FT_HT (Training 256x256x16)')
args = parser.parse_args()

# Load the VQ-VAE model
model = VQVAE()
model.load_state_dict(torch.load(args.model_checkpoint, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the data loader for testing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    # Add any other transformations you need here
])

test_dataset = ImagePairDataset(root_dir_label=args.dataset, root_dir_img=args.data_path, transform=transform)
batch_size = 2  # Modify the batch size based on your needs
num_workers = 4  # Modify the number of workers based on your system's capabilities

data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Initialize lists to store metrics results and corresponding filenames
mae_out_list = []
mse_out_list = []
ssim_out_list = []
psnr_out_list = []
filenames_out_list = []

mae_img_list = []
mse_img_list = []
ssim_img_list = []
psnr_img_list = []
filenames_img_list = []

# Initialize lists to store "bad performance" image filenames and corresponding metrics
bad_performance_filenames = []
bad_performance_metrics = []

# Test the model on the dataset
with torch.no_grad():
    for i, (img, label) in enumerate(tqdm(data_loader)):
        img = img.type(Tensor)  # Convert the input image to the appropriate data type

        # Ensure the model's weights are on the same device as the input data
        model = model.to(img.device)

        # Forward pass through the model
        out, _ = model(img)

        # Save the reconstructed "out" images with the same filename as "img"
        for j in range(len(img)):
            img_filename = test_dataset.image_list[i * batch_size + j]
            out_filename = os.path.join(args.save_path, os.path.basename(img_filename))
            save_image(out[j].cpu(), out_filename, normalize=False)

            # Calculate metrics between "label" and "out"
            label_img = label[j].cpu().numpy()
            out_img = out[j].cpu().numpy()
            mae_out = np.mean(np.abs(label_img - out_img))
            mse_out = np.mean((label_img - out_img) ** 2)

            # Calculate SSIM with an appropriate win_size
            min_side = min(label_img.shape)
            win_size = min(7, min_side)  # Set win_size to the smaller side or a maximum of 7
            ssim_out = ssim(label_img, out_img, win_size=win_size, data_range=out_img.max() - out_img.min(),
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            psnr_out = psnr(label_img, out_img, data_range=out_img.max() - out_img.min())

            # Calculate metrics between "label" and "img"
            img_img = img[j].cpu().numpy()
            mae_img = np.mean(np.abs(label_img - img_img))
            mse_img = np.mean((label_img - img_img) ** 2)

            # Calculate SSIM with an appropriate win_size for "img"
            min_side_img = min(img_img.shape)
            win_size_img = min(7, min_side_img)  # Set win_size to the smaller side or a maximum of 7
            ssim_img = ssim(label_img, img_img, win_size=win_size_img, data_range=img_img.max() - img_img.min(),
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            psnr_img = psnr(label_img, img_img, data_range=img_img.max() - img_img.min())

            # Append metrics and filenames to lists
            mae_out_list.append(mae_out)
            mse_out_list.append(mse_out)
            ssim_out_list.append(ssim_out)
            psnr_out_list.append(psnr_out)
            filenames_out_list.append(os.path.basename(img_filename))

            mae_img_list.append(mae_img)
            mse_img_list.append(mse_img)
            ssim_img_list.append(ssim_img)
            psnr_img_list.append(psnr_img)
            filenames_img_list.append(os.path.basename(img_filename))

            # Compare metrics for "out" and "img" for "bad performance" identification
            if mae_out > mae_img and mse_out > mse_img and ssim_out < ssim_img and psnr_out < psnr_img:
                bad_performance_filenames.append(os.path.basename(img_filename))
                bad_performance_metrics.append({
                    'Filename': os.path.basename(img_filename),
                    'MAE_out': mae_out,
                    'MSE_out': mse_out,
                    'SSIM_out': ssim_out,
                    'PSNR_out': psnr_out,
                    'MAE_img': mae_img,
                    'MSE_img': mse_img,
                    'SSIM_img': ssim_img,
                    'PSNR_img': psnr_img
                })

# Calculate average metrics
average_mae_out = np.mean(mae_out_list)
average_mse_out = np.mean(mse_out_list)
average_ssim_out = np.mean(ssim_out_list)
average_psnr_out = np.mean(psnr_out_list)

average_mae_img = np.mean(mae_img_list)
average_mse_img = np.mean(mse_img_list)
average_ssim_img = np.mean(ssim_img_list)
average_psnr_img = np.mean(psnr_img_list)

# Display average metrics results
print("Metrics for 'label' vs. 'out'")
print(f'Average MAE: {average_mae_out:.4f}')
print(f'Average MSE: {average_mse_out:.4f}')
print(f'Average SSIM: {average_ssim_out:.4f}')
print(f'Average PSNR: {average_psnr_out:.4f}')
print()
print("Metrics for 'label' vs. 'img'")
print(f'Average MAE: {average_mae_img:.4f}')
print(f'Average MSE: {average_mse_img:.4f}')
print(f'Average SSIM: {average_ssim_img:.4f}')
print(f'Average PSNR: {average_psnr_img:.4f}')

# Save metrics results to CSV files
metrics_out_df = pd.DataFrame({
    'Filename': filenames_out_list,
    'MAE': mae_out_list,
    'MSE': mse_out_list,
    'SSIM': ssim_out_list,
    'PSNR': psnr_out_list
})

metrics_img_df = pd.DataFrame({
    'Filename': filenames_img_list,
    'MAE': mae_img_list,
    'MSE': mse_img_list,
    'SSIM': ssim_img_list,
    'PSNR': psnr_img_list
})

metrics_out_csv_path = os.path.join(args.save_path, f'test REST (label vs. out)_(10 epoch).csv')
metrics_img_csv_path = os.path.join(args.save_path, f'test REST (label vs. img)_(10 epoch).csv')

metrics_out_df.to_csv(metrics_out_csv_path, index=False)
metrics_img_df.to_csv(metrics_img_csv_path, index=False)

# Save average metrics results to a separate CSV file
average_metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'SSIM', 'PSNR'],
    'Label vs. Out': [average_mae_out, average_mse_out, average_ssim_out, average_psnr_out],
    'Label vs. Img': [average_mae_img, average_mse_img, average_ssim_img, average_psnr_img]
})

average_metrics_csv_path = os.path.join(args.save_path, f'Metrics_Averages_(10 epoch).csv')
average_metrics_df.to_csv(average_metrics_csv_path, index=False)

# Save "bad performance" image filenames and corresponding metrics to CSV
bad_performance_csv_path = os.path.join(args.save_path, f'Bad_Performance_Images_(10 epoch).csv')
bad_performance_df = pd.DataFrame(bad_performance_metrics)
bad_performance_df.to_csv(bad_performance_csv_path, index=False)
