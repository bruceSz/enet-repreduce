

from data.camvid import   CamVid


from models.enet import ENet
import torch
import torch.optim as optim
import torchvision.transforms  as T
import torch.utils.data as data
import torch.functional as F
#from train import Train
from PIL import Image
import torch.nn as nn

from torch.autograd import Variable
import transforms


class Train():
    def __init__(self, model, data_loader, opti, crit, use_cuda):
        self.model = model
        self.data_loader = data_loader
        self.opti= opti
        self.crit = crit
        self.use_cuda = use_cuda


    def run_epoch(self):
        epoch_loss = 0.0
        for step, batch  in enumerate(self.data_loader):
            inputs, labels = batch

            inputs, labels = Variable(inputs), Variable(labels)

            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = self.model(inputs)

            print("shape of outputs: ", outputs.shape)
            print("shape of labels: ", labels.shape)

            loss = self.crit(outputs, labels)

            self.opti.zero_grad()
            loss.backward()
            self.opti.step()

            epoch_loss += loss.data[0]
            print("Step %d loss: %f" %(step, loss.data[0]))

        return epoch_loss / len(self.data_loader)


from collections import OrderedDict

n_w = 2
num_classes = 12
batch_size = 4
learning_rate = 5e-4
momentum = 0.9
weight_decay = 2e-4
num_epochs = 300
height = 360
width = 480
use_cuda = True and torch.cuda.is_available()


def get_encoding():
# Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    return color_encoding

image_transform = T.Compose(
        [T.Resize((height, width)),
         T.ToTensor()])

label_transform = T.Compose([
        T.Resize((height, width), Image.NEAREST),
       transforms.PILToLongTensor()
    ])

trainset = CamVid("/home/rick/Work/dataset/camvid/CamVid/", 
    transform=T.ToTensor(), label_transform=label_transform)
#args.dataset_dir,
        
trainloader = data.DataLoader(trainset, batch_size= batch_size, shuffle=True, num_workers=n_w )


encoding = get_encoding()
del encoding["road_marking"]

to_pil = transforms.LongTensorToRGBPIL(encoding)

dataiter  = iter(trainloader)

images, labels = dataiter.next()

print("shape of images: ", images.shape)
print("shape of labels: ", labels.shape)
print("shape of unbind labels:", torch.unbind(labels)[0].shape)
labels_list = [T.ToTensor()(to_pil(t))  for t in torch.unbind(labels)]

color_labels = torch.stack(labels_list)

net = ENet(num_classes)

crit = nn.CrossEntropyLoss()


opti = optim.SGD(net.parameters(),lr = learning_rate, momentum=momentum)

if use_cuda:
    net = net.cuda()
    crit = crit.cuda()

train = Train(net, trainloader, opti, crit, use_cuda)

for epoch in range(num_epochs):
    print("[Epoch: %d]" % epoch )
    epoch_loss = train.run_epoch()
    print("[Epoch: %d]. Loss:" % (epoch, epoch_loss))


print("done training")



