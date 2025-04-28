import os
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def save_latent_representations(model, dataset, batch_size=64, save_dir='latent_representations'):
    """
    Сохраняет латентные представления модели в указанную папку.
    
    :param model: Модель, для которой сохраняются латентные представления.
    :param dataset: Датасет для обработки
    :param batch_size: Размер батча.
    :param save_dir: Папка для сохранения представлений.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Processing batches"):
        with torch.no_grad():
            batch = batch.to('cuda')
            h1_batch = model.encoder(batch)  # Обрабатываем сразу батч

        # Сохраняем каждое представление в батче
        for i, h1 in enumerate(h1_batch):
            np.save(os.path.join(save_dir, f'latent_{batch_idx * batch_size + i}.npy'), h1.cpu().numpy())
    print(f"Латентные представления сохранены в папку: {save_dir}")


# Функция для архивации данных в ZIP
def zip_dataset(input_dir, output_zip):
    """
    Архивирует содержимое папки в ZIP-архив.

    :param input_dir: Путь к папке с датасетом.
    :param output_zip: Имя создаваемого ZIP-архива.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Архив создан: {output_zip}")

class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):
        """
        Конструктор датасета.

        :param latent_dir: Путь к папке с файлами .npy (латентными представлениями).
        """
        self.latent_dir = latent_dir
        # Собираем список всех .npy файлов в папке и сортируем их
        self.latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        """
        Возвращает количество элементов в датасете.
        """
        return len(self.latent_files)

    def __getitem__(self, idx):
        """
        Загружает латентное представление по индексу.

        :param idx: Индекс элемента.
        :return: Тензор с латентным представлением.
        """

        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).float()  # Тип float для совместимости с PyTorch
        if self.transform:
            latent = self.transform(latent)

        return latent
