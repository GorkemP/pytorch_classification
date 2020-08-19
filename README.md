# PyTorch Classification
 Minimal and clean training and evaluation codes for baseline performances.
 
## Code Features
* Learning Rate Scheduling is implemented with [torch.optim.lr_Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)  
* [Tensorboard](https://www.tensorflow.org/tensorboard) visualization is added
* Early stopping is implemented
* Best parameters for validation accuracy is saved
* Confusion matrix for validation set is generated
* [Optuna](https://optuna.org/) hyper parameter search framework is used to find best parameters

![Confusion Matrix](img/cm_resnet18.png)

### Training and Validation Sets Accuracies for ResNet18
![accuracies](img/accuracies.svg)

