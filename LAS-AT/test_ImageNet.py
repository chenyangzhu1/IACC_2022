import torchvision
from torch.utils.data import DataLoader

from autoattack import AutoAttack
from LAS_PGD_AT.ImageNet_models import *
from utils import *
import argparse
import sys
import os

sys.path.insert(0, '..')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./cifar-data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str,
                        default='/home/zcaoyao/DLcomp/LAS-AT-master/weights/CIFAR10/LAS-AT/model.pth')
    parser.add_argument('--model', default='WideResNet', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default='./data')
    parser.add_argument('--normalize', type=bool, default=False)
    arguments = parser.parse_args()
    return arguments


args = get_args()
our_mean = (0.485, 0.456, 0.406)
our_std = (0.229, 0.224, 0.225)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
logfile1 = os.path.join(args.out_dir, 'log_file1.txt')
logfile2 = os.path.join(args.out_dir, 'log_file2.txt')
if os.path.exists(logfile1):
    os.remove(logfile1)
if os.path.exists(logfile2):
    os.remove(logfile2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model == "VGG":
    target_model = VGG('VGG19')
elif args.model == "ResNet18":
    target_model = ResNet18()
elif args.model == "PreActResNest18":
    target_model = PreActResNet18()
elif args.model == "WideResNet":
    target_model = WideResNet()
target_model = torch.nn.DataParallel(target_model).cuda()

checkpoint = torch.load(args.model_path)
from collections import OrderedDict

try:
    target_model.load_state_dict(checkpoint)
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    target_model.load_state_dict(new_state_dict, False)
# torch.save(target_model.state_dict(), '/apdcephfs/private_xiaojunjia/LAS-AT/weights/LAS-AT/CIFAR10/LAS_AWP_Trades/model.pth')
if args.normalize == True:
    target_model = nn.Sequential(Normalize(mean=our_mean, std=our_std), target_model)

target_model.eval()
# def MyNew_ImageNet_get_loaders_64(dir_, batch_size):
#     # transform_train = transforms.Compose([
#     #     transforms.Lambda(lambda x: x.convert("RGB")),
#     #     transforms.ToTensor(),
#     # ])
#     #
#     # transform_test = transforms.Compose([
#     #     transforms.Lambda(lambda x: x.convert("RGB")),
#     #     transforms.ToTensor(),
#     # ])
#     # trainset = TinyImageNet(dir_, 'train', transform=transform_train, in_memory=True)
#     # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
#     #
#     #
#     #
#     # testset = TinyImageNet(dir_, 'val', transform=transform_test, in_memory=True)
#     # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
#     #
#     # return trainloader, testloader
#
#
#
#     transform_train = transforms.Compose([
#         transforms.Lambda(lambda x: x.convert("RGB")),
#         transforms.ToTensor(),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.Lambda(lambda x: x.convert("RGB")),
#         transforms.ToTensor(),
#     ])
#
#
#
#
#     data_transform = transforms.Compose([
#         transforms.Resize(32),
#         transforms.CenterCrop(32),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225])
#     ])
#
#     # cifar trans
#     # train_transform = transforms.Compose([
#     #     transforms.RandomCrop(32, padding=4),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #          std=[0.229, 0.224, 0.225])
#     # ])
#     # test_transform = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #          std=[0.229, 0.224, 0.225]),
#     # ])
#
#
#     train_dataset = torchvision.datasets.ImageFolder(
#         root="/mnt/data_zcy/imagenet/ILSVRC2012_img_train/",
#         transform=data_transform)
#     train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
#
#     val_dataset = torchvision.datasets.ImageFolder(
#         root="/mnt/data_zcy/imagenet/ILSVRC2012_img_val/",
#         transform=data_transform)
#
#     val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
#     return train_dataset_loader, val_dataset_loader
#
#
# train_loader, test_loader = MyNew_ImageNet_get_loaders_64(args.data_dir, args.batch_size)
#

def get_test_loader(dir_, batch_size):
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    val_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/data_zcy/imagenet/ILSVRC2012_img_val/",
        transform=data_transform)

    val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return val_dataset_loader


test_loader=get_test_loader(args.data_dir,args.batch_size)


epsilon = args.epsilon
epsilon = float(epsilon) / 255.
print(epsilon)
AT_fgsm_loss,AT_fgsm_acc=evaluate_fgsm(test_loader, target_model, 1)
AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, target_model, 10, 1, epsilon / std)
AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, target_model, 20, 1, epsilon / std)
AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, target_model, 50, 1, epsilon / std)

AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, target_model, 20, 1)


AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, target_model)

print('AT_models_test_acc:', AT_models_test_acc)
print('AT_fgsm_acc:', AT_fgsm_acc)
print('AT_pgd_acc_10:', AT_pgd_acc_10)
print('AT_pgd_acc_20:', AT_pgd_acc_20)
print('AT_pgd_acc_50:', AT_pgd_acc_50)
print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)

adversary1 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile1)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

adv_complete = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
