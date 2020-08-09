import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import optuna
from datetime import datetime

num_epoch = 200
batch_size = 128
best_threshold = 0.0002
num_worker = 4
early_stop_counter = 0
early_stoping_thres = 20
use_multiGPU = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device: ", device)

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        training_loss += loss.item()

    training_loss /= len(train_loader)
    correct /= len(train_loader.dataset)

    return (training_loss, correct)

def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            val_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    val_loss /= len(val_loader)
    correct /= len(val_loader.dataset)

    return (val_loss, correct)

model = models.resnet18(num_classes=10)
# 7x7 convolution is too much for CIFAR where images are 32x32 pixel
model.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

if use_multiGPU:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

model.to(device)

print("Start: " + datetime.now().strftime("%d-%m-%Y (%H:%M)"))

def objective(trial):
    early_stop_counter = 0
    best_acc = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-5, 1), momentum=0.9,
                                weight_decay=trial.suggest_loguniform("wd", 1e-4, 1))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=10, threshold=best_threshold)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        train(model, device, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validation(model, device, val_loader, criterion)
        scheduler.step(val_accuracy)

        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if (val_accuracy > best_acc * (1 + best_threshold)):
            early_stop_counter = 0
            best_acc = val_accuracy
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stoping_thres:
            print("Early stopping at: " + str(epoch))
            break

    return best_acc

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30))
study.optimize(objective, n_trials=20)

print("End: " + datetime.now().strftime("%d-%m-%Y (%H:%M)"))

print("------ Hyperparameter search finished ------")
print("Best parameter: " + str(study.best_trial))

x = [x.params['lr'] for x in study.trials]
y = [x.value for x in study.trials]
plt.xscale("log")
plt.xlim(1e-5, 1)
plt.scatter(x, y)
plt.show()

trials = []
for trial in study.trials:
    x = []
    for c in range(len(trial.intermediate_values)):
        x.append(trial.intermediate_values[c])

    trials.append(x)

import matplotlib.pyplot as plt

for i in range(len(trials)):
    plt.plot(trials[i], label="trial-" + str(i) + ", " + "{:6.5f}".format(study.trials[i].params["lr"]))
plt.legend()
plt.ylim(0, 1)
plt.show()

# If you run this code on jupyter notebook and plotly is installed properly, run the following codes, Optuna provides
# nice visualizations
# optuna.visualization.plot_intermediate_values(study)
# optuna.visualization.plot_optimization_history(study)
