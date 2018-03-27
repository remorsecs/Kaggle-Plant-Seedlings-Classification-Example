import torch
import torch.nn as nn
from models import VGG16
from dataset import PlantSeedlingDataset
from utils import parse_args
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy


args = parse_args()
CUDA_DEVICES = args.cuda_devices
DATASET_ROOT = args.path


def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = PlantSeedlingDataset(Path(DATASET_ROOT).joinpath('train'), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

    model = VGG16(num_classes=train_set.num_classes)
    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.data[0] * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects / len(train_set)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')


if __name__ == '__main__':
    train()
