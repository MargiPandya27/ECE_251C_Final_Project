import argparse
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Models
import models.densenet as dn
import models.vgg as vgg
import models.wcnn as wcnn
import models.dawn as dawn
import models.resnet as resnet

# Datasets utils
from datasets.dtd_config import ImageFilelist
from datasets.transforms import GCN, UnNormalize, Lighting

# try tensorboard
try:
    from tensorboard_logger import configure, log_value
except:
    print("Ignore tensorboard logger")

parser = argparse.ArgumentParser(description='PyTorch Evaluation Only')

# generic training args (some reused for eval)
parser.add_argument('--drop', default=10, type=int, help='drop learning rate (unused in eval)')
parser.add_argument('--epochs', default=0, type=int, help='dummy (not used in eval)')
parser.add_argument('--start-epoch', default=0, type=int, help='dummy (not used in eval)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='dummy (not used in eval)')
parser.add_argument('--momentum', default=0.9, type=float, help='dummy (not used in eval)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='dummy (not used in eval)')
parser.add_argument('--print-freq', '-p', default=50, type=int, help='print frequency')
parser.add_argument('--name', default='eval_run', type=str, help='name of experiment')
parser.add_argument('--tensorboard', action='store_true', help='Log progress to TensorBoard')
parser.add_argument('--multigpu', action='store_true', help='eval using multiple GPUs')
parser.add_argument('--database', choices=['cifar-10','cifar-100','stl','svhn','kth','kth-mix','dtd','imagenet','imagenet_ECCV'],
                    required=True, help='dataset')
parser.add_argument('--num_clases', default=11, type=int, help='num of classes, only works for kth dataset')
parser.add_argument('--summary', default=False, action='store_true', help='print model summary')
parser.add_argument('--gcn', default=False, action='store_true')
parser.add_argument('--lrdecay', nargs='+', type=int, default=[30, 60])
parser.add_argument('--tempdir', type=str, default='/tmp/data', help='where to download CIFAR, etc.')
parser.add_argument('--split_data', default=0, type=int, help='take a limited dataset for eval (CIFAR)')
parser.add_argument('--traindir', default='dataset/kth/train', type=str, help='KTH / ImageNet train dir')
parser.add_argument('--valdir', default='dataset/kth/test', type=str, help='KTH / ImageNet val dir')
parser.add_argument('--monkey', default=False, action='store_true')
parser.add_argument('--pretrained', default=False, action='store_true')

# eval-only flags
parser.add_argument('--eval_only', action='store_true', help='run only evaluation')
parser.add_argument('--eval_ckpt', default='', type=str, help='checkpoint path for evaluation')

# subparsers for models
subparsers = parser.add_subparsers(dest="model")

# densenet
parser_densenet = subparsers.add_parser('densenet')
parser_densenet.add_argument('--layers', default=100, type=int)
parser_densenet.add_argument('--growth', default=12, type=int)
parser_densenet.add_argument('--droprate', default=0, type=float)
parser_densenet.add_argument('--reduce', default=1.0, type=float)
parser_densenet.add_argument('--no-bottleneck', dest='bottleneck', action='store_false')
parser_densenet.set_defaults(bottleneck=True)
parser_densenet.add_argument('--no_init_conv', default=False, action='store_true')

# wcnn
parser_wcnn = subparsers.add_parser('wcnn')
parser_wcnn.add_argument("--wavelet", choices=['haar', 'db2', 'lifting'])
parser_wcnn.add_argument("--levels", default=4, type=int)

# dawn
parser_dwnn = subparsers.add_parser('dawn')
parser_dwnn.add_argument("--regu_details", default=0.1, type=float)
parser_dwnn.add_argument("--regu_approx", default=0.1, type=float)
parser_dwnn.add_argument("--levels", default=4, type=int)
parser_dwnn.add_argument("--first_conv", default=32, type=int)
parser_dwnn.add_argument("--classifier", default='mode1', choices=['mode1','mode2','mode3'])
parser_dwnn.add_argument("--kernel_size", type=int, default=3)
parser_dwnn.add_argument("--no_bootleneck", default=False, action='store_true')
parser_dwnn.add_argument("--share_weights", default=False, action='store_true')
parser_dwnn.add_argument("--simple_lifting", default=False, action='store_true')
parser_dwnn.add_argument("--haar_wavelet", default=False, action='store_true')
parser_dwnn.add_argument('--warmup', default=False, action='store_true')

# resnet
parser_resnet = subparsers.add_parser('resnet')
parser_resnet.add_argument("--use_normal", default=False, action='store_true')
parser_resnet.add_argument("--size_normal", default=3, type=int)
parser_resnet.add_argument("--levels", default=4, type=int)

# scatter (not commonly used for KTH, but kept for completeness)
parser_scatter = subparsers.add_parser('scatter')
parser_scatter.add_argument('--scat', default=2, type=int)
parser_scatter.add_argument('--N', default=32, type=int)
parser_scatter.add_argument('--classifier', type=str, default='WRN')
parser_scatter.add_argument('--mode', type=int, default=1)
parser_scatter.add_argument('--blocks', type=int, default=2)
parser_scatter.add_argument('--use_avg_pool', default=False, action='store_true')

# vgg
parser_vgg = subparsers.add_parser('vgg')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.tensorboard:
        configure("runs/%s" % (args.name))

    print("Eval-only script launched")
    print("Args:", sys.argv)

    USE_COLOR = not args.monkey
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # -------------------------
    # Dataset / dataloaders
    # -------------------------
    if args.database == 'kth':
        if not USE_COLOR and args.gcn:
            raise RuntimeError("It is not possible to use grayimage and GCN")

        if args.gcn:
            normalize = GCN()
        else:
            if USE_COLOR:
                normalize = transforms.Normalize(
                    mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            else:
                normalize = transforms.Normalize(
                    mean=[x/255.0 for x in [125.3]],
                    std=[x/255.0 for x in [63.0]])

        add_transform = []
        if not USE_COLOR:
            add_transform += [transforms.Grayscale(num_output_channels=1)]

        transform_train = transforms.Compose(
            add_transform + [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        transform_test = transforms.Compose(
            add_transform + [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

        # Using ImageFolder, traindir/valdir must exist
        kth_train_dataset = datasets.ImageFolder(root=args.traindir,
                                                 transform=transform_train)
        kth_test_dataset = datasets.ImageFolder(root=args.valdir,
                                                transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            kth_train_dataset, shuffle=True,
            batch_size=args.batch_size, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            kth_test_dataset, shuffle=True,
            batch_size=args.batch_size, **kwargs)

        NUM_CLASS = args.num_clases
        INPUT_SIZE = 224

    elif args.database == 'cifar-10':
        if not USE_COLOR:
            raise RuntimeError("CIFAR-10 does not handle gray images")

        normalize = transforms.Normalize(
            mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        data_CIFAR10 = datasets.CIFAR10(args.tempdir, train=True, download=True,
                                        transform=transform_train)
        if args.split_data > 0:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=[1] * 10000, num_samples=args.split_data)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            data_CIFAR10,
            batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.tempdir, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        NUM_CLASS = 10
        INPUT_SIZE = 32

    elif args.database == 'cifar-100':
        if not USE_COLOR:
            raise RuntimeError("CIFAR-100 does not handle gray images")

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        data_CIFAR100 = datasets.CIFAR100(args.tempdir, train=True, download=True,
                                          transform=transform_train)
        if args.split_data > 0:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=[1] * 10000, num_samples=args.split_data)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            data_CIFAR100,
            batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.tempdir, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        NUM_CLASS = 100
        INPUT_SIZE = 32

    else:
        raise RuntimeError("Eval-only script currently wired for kth / cifar-10 / cifar-100")

    if not USE_COLOR and args.model != "dawn":
        raise RuntimeError("Only DAWN supports gray images")

    # -------------------------
    # Create model
    # -------------------------
    big_input = INPUT_SIZE != 32
    scattering = None

    if args.model == 'densenet':
        no_init_conv = args.no_init_conv
        if INPUT_SIZE > 128:
            no_init_conv = False
        model = dn.DenseNet3(
            args.layers, NUM_CLASS, args.growth,
            reduction=args.reduce,
            bottleneck=args.bottleneck,
            dropRate=args.droprate,
            init_conv=not no_init_conv)

    elif args.model == 'vgg':
        model = vgg.VGG(NUM_CLASS, big_input=big_input)

    elif args.model == 'wcnn':
        model = wcnn.WCNN(NUM_CLASS, big_input=big_input,
                          wavelet=args.wavelet, levels=args.levels)

    elif args.model == 'dawn':
        model = dawn.DAWN(
            NUM_CLASS, big_input=big_input,
            first_conv=args.first_conv,
            number_levels=args.levels,
            kernel_size=args.kernel_size,
            no_bootleneck=args.no_bootleneck,
            classifier=args.classifier,
            share_weights=args.share_weights,
            simple_lifting=args.simple_lifting,
            COLOR=USE_COLOR,
            regu_details=args.regu_details,
            regu_approx=args.regu_approx,
            haar_wavelet=args.haar_wavelet
        )

    elif args.model == 'resnet':
        if big_input:
            import torchvision
            model = torchvision.models.resnet18(pretrained=args.pretrained)
            model.fc = nn.Linear(512, NUM_CLASS)
        else:
            if args.use_normal:
                model = resnet.ResNetCIFARNormal(
                    [args.size_normal, args.size_normal, args.size_normal],
                    num_classes=NUM_CLASS)
            else:
                model = resnet.ResNetCIFAR(
                    [2, 2, 2, 2], num_classes=NUM_CLASS, levels=args.levels)

    elif args.model == 'scatter':
        from kymatio import Scattering2D
        from models.scatter.Scatter_WRN import Scattering2dCNN, ScatResNet

        if INPUT_SIZE == 224:
            scattering = Scattering2D(J=args.scat, shape=(args.N, args.N), max_order=args.mode)
            scattering = scattering.cuda()
            model = ScatResNet(args.scat, INPUT_SIZE, NUM_CLASS,
                               args.classifier, args.mode)
        else:
            scattering = Scattering2D(J=args.scat, shape=(args.N, args.N), max_order=args.mode)
            scattering = scattering.cuda()
            model = Scattering2dCNN(
                args.classifier, J=args.scat, N=args.N,
                num_classes=NUM_CLASS, blocks=args.blocks,
                mode=args.mode, use_avg_pool=args.use_avg_pool)
    else:
        raise RuntimeError("Unknown model")

    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
    print("Number of trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.multigpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    if args.summary:
        from torchsummary import summary
        summary(model, input_size=(3, INPUT_SIZE, INPUT_SIZE))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    # -------------------------
    # EVAL ONLY
    # -------------------------
    if not args.eval_only:
        print("WARNING: eval_only flag not set; running eval anyway.")
    if args.eval_ckpt:
      if os.path.isfile(args.eval_ckpt):
          print(f"=> loading checkpoint '{args.eval_ckpt}'")
          checkpoint = torch.load(args.eval_ckpt)
          best_prec1_ckpt = checkpoint.get('best_prec1', 0)

          # Handle DataParallel prefix
          state_dict = checkpoint['state_dict']
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in state_dict.items():
              if k.startswith('module.'):
                  new_state_dict[k[7:]] = v  # remove 'module.' prefix
              else:
                  new_state_dict[k] = v

          # Try strict load; if fails, allow partial
          try:
              model.load_state_dict(new_state_dict, strict=True)
              print(f"=> loaded checkpoint '{args.eval_ckpt}' (epoch {checkpoint.get('epoch', 'unknown')}, best_prec1={best_prec1_ckpt:.3f})")
          except RuntimeError as e:
              print("Warning: strict load failed, attempting partial load...")
              # Partial load
              model_dict = model.state_dict()
              pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
              model_dict.update(pretrained_dict)
              model.load_state_dict(model_dict)
              print(f"=> partially loaded checkpoint '{args.eval_ckpt}' with {len(pretrained_dict)}/{len(model_dict)} layers matched")
      else:
          print(f"=> no checkpoint found at '{args.eval_ckpt}', evaluating randomly initialized model")


    is_dwnn = (args.model == 'dawn')
    prec1_val, prec5_val, loss_val = validate(
        val_loader, model, criterion, epoch=0, is_dwnn=is_dwnn, scattering=scattering)

    print('Evaluation done.')
    print('Prec@1: {:.3f} %, Prec@5: {:.3f} %, Loss: {:.4f}'.format(
        prec1_val, prec5_val, loss_val))


def validate(val_loader, model, criterion, epoch, is_dwnn, scattering=None):
    batch_time = AverageMeter()
    losses_total = AverageMeter()
    losses_class = AverageMeter()
    losses_regu = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            if args.model == 'scatter' and scattering is not None:
                input = scattering(input)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            is_regu_activated = False
            if is_dwnn:
                output, regus = model(input_var)
                loss_class = criterion(output, target_var)
                loss_total = loss_class
                if regus[0]:
                    loss_regu = sum(regus)
                    loss_total += loss_regu
                    is_regu_activated = True
            else:
                output = model(input_var)
                loss_class = criterion(output, target_var)
                loss_total = loss_class

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            if args.num_clases >= 5:
                p_m = 5
            else:
                p_m = 3
            prec5 = accuracy(output.data, target, topk=(p_m,))[0]

            losses_total.update(loss_total.item(), input.size(0))
            if is_regu_activated:
                losses_regu.update(loss_regu.item(), input.size(0))
            else:
                losses_regu.update(0.0, input.size(0))
            losses_class.update(loss_class.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss (Class) {loss_class.val:.4f} ({loss_class.avg:.4f})\t'
                      'Loss (Regu) {loss_regu.val:.4f} ({loss_regu.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss_class=losses_class, loss_regu=losses_regu,
                          top1=top1, top5=top5))

    if args.tensorboard:
        log_value('val_loss', losses_total.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return (top1.avg, top5.avg, losses_total.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
