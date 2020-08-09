import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import time

learning_rate = 0.05
num_epoch = 100
best_acc = 0
batch_size = 32
num_worker = 8
log_interval = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Data loading code

traindir = "/home/gorkem/Desktop/data/IMAGENET/train/train"
valdir = "/home/gorkem/Desktop/data/IMAGENET/val/val"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])
train_dataset = datasets.ImageFolder(traindir, train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                           pin_memory=True)

val_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
val_dataset = datasets.ImageFolder(valdir, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                                         pin_memory=True)

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    batch_time = AverageMeter('Batch time', ':6.3f')
    data_time = AverageMeter('Data Load time', ':6.3f')

    running_loss = 0.0
    top1 = 0
    top5 = 0
    final_batch = 0

    end = time.time()
    for batch_id, (data, target) in enumerate(train_loader):
        if batch_id != 0:
            data_time.update(time.time() - end)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        output = model(data)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        top1 += acc1.item()
        top5 += acc5.item()
        final_batch = batch_id

        if batch_id != 0:
            batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % 10 == 0:
            print(str(batch_id * batch_size) + ": " + str(batch_time) + " | " + str(data_time))

    return running_loss / len(train_loader), final_batch, top1 / len(train_loader), top5 / len(
            train_loader)

def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            val_loss += criterion(output, target).item()
            top1 += acc1.item()
            top5 += acc5.item()

    val_loss /= len(val_loader)
    top1 /= len(val_loader)
    top5 /= len(val_loader)

    return val_loss, top1, top5

def main():
    model = models.resnet18()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)

    experiment_signature = model._get_name() + " lr=" + str(learning_rate) + " bs=" + str(batch_size) + " epoch=" + str(
            num_epoch) + " num_worker=" + str(num_worker)
    print("model: " + experiment_signature)
    # writer = SummaryWriter('runs/' + experiment_signature + " t=" + datetime.now().strftime("%d-%m-%Y (%H:%M)"))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        adjust_learning_rate(optimizer, epoch)

        start = time.time()
        train_loss, total_batch, train_top1, train_top5 = train(model, device, train_loader, criterion, optimizer)
        elapsed = time.time() - start

        val_loss, val_top1, val_top5 = validation(model, device, val_loader, criterion)

        print("epoch: {:3.0f}".format(epoch + 1) + " | time: {:3.0f} sec".format(
                elapsed) + " | Average batch-process time: {:4.3f} sec".format(elapsed / total_batch))

        if (val_top1 > best_acc):
            best_acc = val_top1
            print("overwriting the best model!")
            torch.save(model.state_dict(), model._get_name() + '_checkpoint_best.pth.tar')

    # writer.close()
    print("------ Training finished ------")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(optimizer, epoch):
    if epoch == 50:
        print("learning_rate decreased")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
    elif epoch == 100:
        print("learning_rate decreased")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
