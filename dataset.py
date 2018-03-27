from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class PlantSeedlingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = 0

        if self.root_dir.name == 'train':
            for i, _dir in enumerate(self.root_dir.glob('*')):
                for file in _dir.glob('*'):
                    self.x.append(file)
                    self.y.append(i)

                self.num_classes += 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
