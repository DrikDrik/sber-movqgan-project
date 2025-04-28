import torchvision.transforms as transforms
from utils import download_datasets
from utils.celeba import CelebADataset as CelebA

utils.download_datasets.main()
class CelebaCustomDataset(CelebA):
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image

def get_dataset():
  dataset = CelebaCustomDataset(
    transform=transforms.Compose([
      transforms.Resize(178),
      transforms.CenterCrop(178),
      transforms.ToTensor(),
      lambda x: x * 2 - 1]),
    root_dir='/content/')
  return dataset
  
  
