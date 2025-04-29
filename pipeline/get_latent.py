import os
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def save_latent_representations(model, dataset, batch_size=64, save_dir='latent_representations'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Processing batches"):
        with torch.no_grad():
            batch = batch.to('cuda')
            h1_batch = model.encoder(batch)  # Обрабатываем сразу батч

        for i, h1 in enumerate(h1_batch):
            np.save(os.path.join(save_dir, f'latent_{batch_idx * batch_size + i}.npy'), h1.cpu().numpy())



def zip_dataset(input_dir, output_zip):

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Архив создан: {output_zip}")

class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):

        self.latent_dir = latent_dir
        # Собираем список всех .npy файлов в папке и сортируем их
        self.latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):

        return len(self.latent_files)

    def __getitem__(self, idx):


        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).float()  # Тип float для совместимости с PyTorch
        if self.transform:
            latent = self.transform(latent)

        return latent
