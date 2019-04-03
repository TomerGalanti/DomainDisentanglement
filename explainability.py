import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from utils import ImageFolder

import os
import argparse

from models import Classifier2, Classifier, E_common, E_separate_A, E_separate_B, Decoder
from utils_classifier import progress_bar
from utils import get_train_dataset, get_test_dataset
from utils import save_imgs, save_model, load_model_for_eval, save_stripped_imgs, save_imgs_explainability

import pdb


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume_encoders', default='False')
parser.add_argument('--root', default='')
parser.add_argument('--test_root', default='')
parser.add_argument('--out', default='out')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--iters', type=int, default=1250000)
parser.add_argument('--resize', type=int, default=128)
parser.add_argument('--crop', type=int, default=178)
parser.add_argument('--sep', type=int, default=25)
parser.add_argument('--discweight', type=float, default=0.001)
parser.add_argument('--disclr', type=float, default=0.0002)
parser.add_argument('--progress_iter', type=int, default=100)
parser.add_argument('--display_iter', type=int, default=1000)
parser.add_argument('--log_iter', type=int, default=100)
parser.add_argument('--save_iter', type=int, default=10000)
parser.add_argument('--load', default='')
parser.add_argument('--iter', default=10000, type=int)
parser.add_argument('--num_display', type=int, default=12)
parser.add_argument('--zeroweight', type=float, default=1.0)
parser.add_argument('--reconweight', type=float, default=0.01)
parser.add_argument('--imgdisc', type=float, default=0)  # )0.001)
parser.add_argument('--elastic', type=int, default=0)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainA, trainB = get_train_dataset(args)  # torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloaderA = torch.utils.data.DataLoader(trainA, batch_size=args.bs, shuffle=True, num_workers=2)
trainloaderB = torch.utils.data.DataLoader(trainB, batch_size=args.bs, shuffle=True, num_workers=2)

testA, testB = get_test_dataset(args)
testloaderA = torch.utils.data.DataLoader(testA, batch_size=args.bs, shuffle=False, num_workers=2)
testloaderB = torch.utils.data.DataLoader(testB, batch_size=args.bs, shuffle=False, num_workers=2)

print(len(testloaderA))


#pdb.set_trace()



classes = ('A', 'B')

size = args.resize // 64
common_dim = (512 - 2 * args.sep) * size * size
sepA_dim = sepB_dim = args.sep * size * size


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
#net = Classifier2(0, args.resize // 64)
net = Classifier(common_dim, sepA_dim, sepB_dim)
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
e_common = E_common(args.sep, size)
e_separate_A = E_separate_A(args.sep, size)
e_separate_B = E_separate_B(args.sep, size)
decoder = Decoder(size)

net = net.to(device)
e_common = e_common.to(device)
e_separate_A = e_separate_A.to(device)
e_separate_B = e_separate_B.to(device)
decoder = decoder.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.isdir('./classifier/' + args.root):
    os.mkdir('./classifier/' + args.root)

if not os.path.isdir('./classifier/' + args.root + '/checkpoint/'):
    os.mkdir('./classifier/' + args.root + '/checkpoint/')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./classifier/' + args.root + '/checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.resume_encoders == 'True':
    # Load checkpoint.
    print('==> Resuming from checkpoint of encodings..')
    save_file = os.path.join(args.load, 'checkpoint'+str(args.iter))
    _iter = load_model_for_eval(save_file, e_common, e_separate_A, e_separate_B, decoder, )


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train_step(inputs, targets, train_loss, total, correct, batch_idx, trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()

    if args.resume_encoders == '':
        outputs = net(inputs)
    elif args.resume_encoders == 'True':
        common = e_common(inputs)
        A_separate = e_separate_A(inputs)
        B_separate = e_separate_B(inputs)
        encoding = torch.cat([common, A_separate, B_separate], dim=1)
        outputs = net(encoding)
    else:
        print('Error: undefined resume_encoders')
        exit()

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss, total, correct


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputsA, inputsB) in enumerate(zip(trainloaderA, trainloaderB)):
        targetsA = torch.zeros((len(inputsA)), dtype=torch.long).to(device)
        targetsB = torch.ones((len(inputsB)), dtype=torch.long).to(device)

        train_loss, total, correct = train_step(inputsA, targetsA, train_loss, total, correct, batch_idx, trainloaderA)
        train_loss, total, correct = train_step(inputsB, targetsB, train_loss, total, correct, batch_idx, trainloaderB)


def test_step(inputs, targets, test_loss, total, correct, batch_idx, testloader):
    inputs, targets = inputs.to(device), targets.to(device)

    if args.resume_encoders == '':
        outputs = net(inputs)
    elif args.resume_encoders == 'True':
        common = e_common(inputs)
        A_separate = e_separate_A(inputs)
        B_separate = e_separate_B(inputs)
        encoding = torch.cat([common, A_separate, B_separate], dim=1)
        outputs = net(encoding)
    else:
        print('Error: undefined resume_encoders')
        exit()

    loss = criterion(outputs, targets)

    test_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return test_loss, total, correct


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputsA, inputsB) in enumerate(zip(testloaderA, testloaderB)):
            targetsA = torch.zeros((len(inputsA)), dtype=torch.long).to(device)
            targetsB = torch.ones((len(inputsB)), dtype=torch.long).to(device)

            test_loss, total, correct = test_step(inputsA, targetsA, test_loss, total, correct, batch_idx, testloaderA)
            test_loss, total, correct = test_step(inputsB, targetsB, test_loss, total, correct, batch_idx, testloaderB)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './classifier/' + args.root + '/checkpoint/ckpt.t7')
        best_acc = acc

    save_imgs_explainability(args, net, e_common, e_separate_A, e_separate_B, decoder, epoch, A=True)
    save_imgs_explainability(args, net, e_common, e_separate_A, e_separate_B, decoder, epoch, A=True)


def test_on_data(params):
    comp_transform = transforms.Compose([
        transforms.Resize(params.resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_test = ImageFolder(params.test_root, transform=comp_transform)
    testloaderA = torch.utils.data.DataLoader(domA_test, batch_size=args.bs, shuffle=True, num_workers=2)

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputsA in enumerate(testloaderA):
            targetsA = torch.zeros((len(inputsA)), dtype=torch.long).to(device)

            test_loss, total, correct = test_step(inputsA, targetsA, test_loss, total, correct, batch_idx, testloaderA)

    # # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './classifier/' + args.root + '/checkpoint/ckpt.t7')
    #     best_acc = acc


if args.test_root:
    test_on_data(args)
else:
    for epoch in range(start_epoch, start_epoch + 200):
        print('train')
        train(epoch)
        print('test')
        test(epoch)


