import argparse
from typing import Callable, Tuple

import optuna  # type: ignore
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from torchvision.datasets import MNIST  # type: ignore


def create_loader(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


class Net(nn.Module):

    def __init__(self, trial: optuna.trial.Trial) -> None:
        super(Net, self).__init__()
        self.activation = getattr(F, trial.suggest_categorical('activation', ['relu', 'elu']))
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(p=trial.suggest_uniform("dropout_prob1", 0.1, 0.9))
        self.dropout2 = nn.Dropout2d(p=trial.suggest_uniform("dropout_prob2", 0.1, 0.9))
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model: nn.Module, device: str, train_loader: DataLoader, optimizer: optim.Optimizer) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model: nn.Module, device: str, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 1 - correct / len(test_loader.dataset)


def get_optimizer(trial: optuna.trial.Trial, model: nn.Module) -> optim.Optimizer:

    def adam(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def momentum(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'momentum'])
    optimizer: Callable[[nn.Module, float, float], optim.Optimizer] = locals()[optimizer_name]
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    return optimizer(model, lr, weight_decay)


def objective_wrapper(train_loader: DataLoader, test_loader: DataLoader,
                      epochs: int) -> Callable[[optuna.trial.Trial], float]:

    def objective(trial: optuna.trial.Trial) -> float:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Net(trial).to(device)
        optimizer = get_optimizer(trial, model)

        for step in range(epochs):
            train(model, device, train_loader, optimizer)
            error_rate = test(model, device, test_loader)

            trial.report(error_rate, step)
            if trial.should_prune(step):
                raise optuna.exceptions.TrialPruned()

        return error_rate

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchs', type=int, default=128)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--output', default='result.csv')
    args = parser.parse_args()

    train_loader, test_loader = create_loader(args.batchs)
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    objective = objective_wrapper(train_loader, test_loader, args.epochs)
    study.optimize(objective, n_trials=args.trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best error rate: {study.best_value}")

    study.trials_dataframe().to_csv('result.csv')


if __name__ == '__main__':
    main()
