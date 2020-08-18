import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report

from datetime import datetime
import time
from utils import plot_confusion_matrix, initialize_model

torch.manual_seed(35)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

learning_rate = 0.01
num_epoch = 200
best_acc = 0
batch_size = 128
weight_decay = 0.01
best_threshold = 0.0002
num_worker = 4
early_stop_counter = 0
early_stoping_thres = 20
use_multiGPU = False
enable_tensorboard = True
pretrained_weights = True

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

model = initialize_model("VGG", pretrained_weights, 10)

if use_multiGPU:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

model.to(device)

experiment_signature = model.__class__.__name__ + " lr=" + str(learning_rate) + " bs=" + str(
        batch_size) + " reg=" + str(weight_decay)
print(experiment_signature)

if enable_tensorboard:
    writer = SummaryWriter('runs/' + experiment_signature + " t=" + datetime.now().strftime("%d-%m-%Y (%H:%M)"))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=10, threshold=best_threshold,
                                           verbose=True)
last_epoch = 0
for epoch in range(num_epoch):
    last_epoch = epoch

    start = time.time()
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
    elapsed = time.time() - start

    val_loss, val_accuracy = validation(model, device, val_loader, criterion)
    scheduler.step(val_accuracy)

    print("epoch: {:3.0f}".format(epoch + 1) + " | time: {:3.0f} sec".format(
            elapsed) + " | Average batch-process time: {:4.3f} sec".format(
            elapsed / len(train_loader)) + " | Train acc: {:4.2f}".format(
            train_accuracy * 100) + " | Val acc: {:4.2f}".format(
            val_accuracy * 100))

    if enable_tensorboard:
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalars('Losses', {'Train': train_loss, 'Val': val_loss}, epoch + 1)
        writer.add_scalars('Accuracies', {'Train': train_accuracy, 'Val': val_accuracy}, epoch + 1)

    if (val_accuracy > best_acc * (1 + best_threshold)):
        early_stop_counter = 0
        best_acc = val_accuracy
        print("overwriting the best model!")
        torch.save(model.state_dict(), model._get_name() + '_checkpoint_best.pth.tar')
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stoping_thres:
        print("Early stopping at: " + str(epoch))
        break

if enable_tensorboard:
    writer.add_hparams({'model': model.__class__.__name__, 'lr': learning_rate, 'batch size': batch_size,
                        "epoch": last_epoch, "optimizer": optimizer.__class__.__name__},
                       {'accuracy': best_acc * 100})

if enable_tensorboard:
    writer.close()

print("------ Training finished ------")
print("Best validation set accuracy: " + str(best_acc * 100))
print("plotting confusion matrix...")
# Results on validation set

validation_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_worker)
model.load_state_dict(torch.load(model._get_name() + '_checkpoint_best.pth.tar'))
model.to(device)
model.eval()

y_true = []
y_pred = []
with torch.no_grad():
    for data, target in validation_loader:
        data = data.to(device)
        y_true.append(target.item())

        output = model(data)
        prediction = output.argmax(dim=1, keepdim=True)[0][0].item()
        y_pred.append(prediction)

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred, target_names=train_set.classes)

plot_confusion_matrix(cm, train_set.classes, "confusion matrix", normalize=False)
print(cr)
