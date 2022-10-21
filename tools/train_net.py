import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


import torch.optim as optim

from tqdm import tqdm


from mdistiller.engine.utils import adjust_learning_rate


class LinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, n_label)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="stl10",
        choices=["cifar100", "imagenet","stl10"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # print("num_classes:",num_classes)
    model, pretrain_model_path = cifar_model_dict[args.model]
    model = model(num_classes=100)
    ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
    model.load_state_dict(load_checkpoint(ckpt)["model"])
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.eval()


    # model = torch.nn.DataParallel(model.cuda())
    net = NonLinearClassifier(128, 10)
    # net = LinearClassifier(128, 10)
    net.cuda()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=cfg.SOLVER.LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=0
    )


    epochs = 40
    best_acc = 0.0
    train_steps = len(train_loader)
    print(train_steps)

    print("start train!")

    for epoch in range(epochs):
        lr = adjust_learning_rate(epoch, cfg, optimizer)
        # train
        net.train()
        running_loss = 0.0
        # train_bar = tqdm(range(train_steps))
        train_bar = tqdm(train_loader)
        for idx, data in enumerate(train_bar):
            optimizer.zero_grad()
            image, target, _ = data
            image = image.float()
            image = image.cuda(non_blocking=True)

            # print("image shape:", image.shape)

            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                _, feature = model(image)

            feature_in = feature["pooled_feat"]
            # print("pooled feature shape:", feature_in.shape)

            logits = net(feature_in)

            # print("logits shape:", logits.shape)
            # print("target shape:", target.shape)
            loss = F.cross_entropy(logits, target)
            # loss = nn.CrossEntropyLoss(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validation

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for (image, target) in val_bar:
            # for idx, (image, target) in enumerate(val_loader):
                image = image.float()
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                _, feature = model(image)
                feature_in = feature["pooled_feat"]
                outputs = net(feature_in)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, target).sum().item()
        # val_acc = acc / (len(train_loader) * cfg.DATASET.TEST.BATCH_SIZE)
        val_acc = acc / num_data

        print('[epoch %d] train_loss: %.3f  val_accu: %.3f' %
              (epoch + 1, running_loss / train_steps, val_acc))



        if val_acc > best_acc:
            best_acc = val_acc

    print('Finished Training')
    print("best_acc:", best_acc)

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
#
# cudnn.benchmark = True
#
# from mdistiller.models import cifar_model_dict, imagenet_model_dict
# from mdistiller.distillers import distiller_dict
# from mdistiller.dataset import get_dataset
# from mdistiller.engine.utils import load_checkpoint, log_msg
# from mdistiller.engine.cfg import CFG as cfg
# from mdistiller.engine.cfg import show_cfg
# from mdistiller.engine import trainer_dict
#
#
# import torch.optim as optim
#
# from tqdm import tqdm
#
#
# from mdistiller.engine.utils import adjust_learning_rate
#
#
# class LinearClassifier(nn.Module):
#
#     def __init__(self, dim_in, n_label=10):
#         super(LinearClassifier, self).__init__()
#
#         self.net = nn.Linear(dim_in, n_label)
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class NonLinearClassifier(nn.Module):
#
#     def __init__(self, dim_in, n_label=10, p=0.1):
#         super(NonLinearClassifier, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(dim_in, 200),
#             nn.Dropout(p=p),
#             nn.BatchNorm1d(200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, n_label),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# def fix_bn(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.eval()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-m", "--model", type=str, default="")
#     parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
#     parser.add_argument(
#         "-d",
#         "--dataset",
#         type=str,
#         default="stl10",
#         choices=["cifar100", "imagenet","stl10"],
#     )
#     parser.add_argument("-bs", "--batch-size", type=int, default=64)
#     args = parser.parse_args()
#
#     cfg.DATASET.TYPE = args.dataset
#     cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
#
#     # cfg & loggers
#     show_cfg(cfg)
#     # init dataloader & models
#     train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
#     # print("num_classes:",num_classes)
#     model, pretrain_model_path = cifar_model_dict[args.model]
#     model = model(num_classes=100)
#     ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
#     model.load_state_dict(load_checkpoint(ckpt)["model"])
#     model = model.cuda()
#     # model = torch.nn.DataParallel(model)
#
#     for param in model.parameters():
#         param.requires_grad = False
#
#     model.apply(fix_bn)
#
#     # classname = model.__class__.__name__
#     # if classname.find('BatchNorm') != -1:
#     #     model.eval()
#
#     num_feats = model.fc.in_features
#     model.fc = nn.Linear(num_feats, 10)
#     for param in model.fc.parameters():
#         param.requires_grad = True
#     model.cuda()
#     optimizer = optim.SGD(
#         model.fc.parameters(),
#         lr=cfg.SOLVER.LR,
#         momentum=cfg.SOLVER.MOMENTUM,
#         weight_decay=0
#     )
#
#
#     epochs = 40
#     best_acc = 0.0
#     train_steps = len(train_loader)
#     print(train_steps)
#
#     print("start train!")
#
#     for epoch in range(epochs):
#         lr = adjust_learning_rate(epoch, cfg, optimizer)
#         # train
#         model.train()
#         running_loss = 0.0
#         # train_bar = tqdm(range(train_steps))
#         train_bar = tqdm(train_loader)
#         for idx, data in enumerate(train_bar):
#
#             optimizer.zero_grad()
#             image, target, _ = data
#             image = image.float()
#             image = image.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#             logits, _ = model(image)
#             loss = F.cross_entropy(logits, target)
#             loss.requires_grad_()
#             loss.backward()
#             optimizer.step()
#             # with torch.set_grad_enabled(True):
#             #     image, target, _ = data
#             #     image = image.float()
#             #     image = image.cuda(non_blocking=True)
#             #     target = target.cuda(non_blocking=True)
#             #     logits, _ = model(image)
#             #     loss = F.cross_entropy(logits, target)
#             #     loss.requires_grad_()
#             #     loss.backward()
#             #     optimizer.step()
#
#             running_loss += loss.item()
#             train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
#                                                                      epochs,
#                                                                      loss)
#         # validation
#
#         model.eval()
#         acc = 0.0
#         with torch.no_grad():
#             val_bar = tqdm(val_loader)
#             for (image, target) in val_bar:
#             # for idx, (image, target) in enumerate(val_loader):
#                 image = image.float()
#                 image = image.cuda(non_blocking=True)
#                 target = target.cuda(non_blocking=True)
#
#                 outputs, _ = model(image)
#                 # feature_in = feature["pooled_feat"]
#                 # outputs = net(feature_in)
#                 predict_y = torch.max(outputs, dim=1)[1]
#                 acc += torch.eq(predict_y, target).sum().item()
#         # val_acc = acc / (len(train_loader) * cfg.DATASET.TEST.BATCH_SIZE)
#         val_acc = acc / num_data
#
#         print('[epoch %d] train_loss: %.3f  val_accu: %.3f' %
#               (epoch + 1, running_loss / train_steps, val_acc))
#
#
#
#         if val_acc > best_acc:
#             best_acc = val_acc
#
#     print('Finished Training')
#     print("best_acc:", best_acc)