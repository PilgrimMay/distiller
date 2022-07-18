import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter

from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)




class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, rank=0, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch, rank)
            epoch += 1
            self.train_loader.sampler.set_epoch(epoch)
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch, rank):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        if rank == 0:
            test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()

        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class MSTrainer(object):
    def __init__(self, experiment_name, distiller1, distiller2, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller1 = distiller1
        self.distiller2 = distiller2
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer1, self.optimizer2 = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer1 = optim.SGD(
                self.distiller1.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            optimizer2 = optim.SGD(
                self.distiller2.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer1, optimizer2

    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        print("-----------------------MultiStudent-----------------------")
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller1.load_state_dict(state["model1"])
            self.distiller2.load_state_dict(state["model2"])
            self.optimizer1.load_state_dict(state["optimizer1"])
            self.optimizer2.load_state_dict(state["optimizer2"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))


    def train_epoch(self, epoch):
        lr1 = adjust_learning_rate(epoch, self.cfg, self.optimizer1)
        lr2 = adjust_learning_rate(epoch, self.cfg, self.optimizer2)
        train_meters_1 = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        train_meters_2 = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar1 = tqdm(range(num_iter))
        pbar2 = tqdm(range(num_iter))

        # train loops
        self.distiller1.train()
        self.distiller2.train()
        for idx, data in enumerate(self.train_loader):
            msg1, msg2 = self.train_iter(data, epoch, train_meters_1, train_meters_2)
            pbar1.set_description(log_msg(msg1, "TRAIN"))
            pbar2.set_description(log_msg(msg2, "TRAIN"))
            pbar1.update()
            pbar2.update()
        pbar1.close()
        pbar2.close()

        # validate
        test_acc_1, test_acc_top5_1, test_loss_1 = validate(self.val_loader, self.distiller1)
        test_acc_2, test_acc_top5_2, test_loss_2 = validate(self.val_loader, self.distiller2)

        # log
        log_dict_1 = OrderedDict(
            {
                "train_acc": train_meters_1["top1"].avg,
                "train_loss": train_meters_1["losses"].avg,
                "test_acc": test_acc_1,
                "test_acc_top5": test_acc_top5_1,
                "test_loss": test_loss_1,
            }
        )
        log_dict_2 = OrderedDict(
            {
                "train_acc": train_meters_2["top1"].avg,
                "train_loss": train_meters_2["losses"].avg,
                "test_acc": test_acc_2,
                "test_acc_top5": test_acc_top5_2,
                "test_loss": test_loss_2,
            }
        )
        self.log(lr1, epoch, log_dict_1)
        self.log(lr2, epoch, log_dict_2)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model1": self.distiller1.state_dict(),
            "model2": self.distiller2.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state_1 = {"model1": self.distiller1.module.student.state_dict()}
        student_state_2 = {"model2": self.distiller2.module.student.state_dict()}

        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state_1, os.path.join(self.log_path, "student1_latest")
        )
        save_checkpoint(
            student_state_2, os.path.join(self.log_path, "student2_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state_1,
                os.path.join(self.log_path, "student_1{}".format(epoch)),
            )
            save_checkpoint(
                student_state_2,
                os.path.join(self.log_path, "student_2{}".format(epoch)),
            )
        # update the best
        if test_acc_1 >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state_1, os.path.join(self.log_path, "student_best")
            )
        if test_acc_2 >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state_2, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters_1, train_meters_2):
        # self.optimizer.zero_grad()

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        train_start_time = time.time()
        image, target, index = data
        train_meters_1["data_time"].update(time.time() - train_start_time)
        train_meters_2["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        batch_size = image.size(0)
        # forward
        preds_1, losses_dict_1 = self.distiller1(image=image, target=target, epoch=epoch)
        preds_2, losses_dict_2 = self.distiller2(image=image, target=target, epoch=epoch)

        # backward student1
        loss1 = sum([l.mean() for l in losses_dict_1.values()])
        loss1.backward()
        # self.optimizer1.step()

        # collect student1 info
        acc1_1, acc5_1 = accuracy(preds_1, target, topk=(1, 5))
        train_meters_1["losses"].update(loss1.cpu().detach().numpy().mean(), batch_size)
        train_meters_1["top1"].update(acc1_1[0], batch_size)
        train_meters_1["top5"].update(acc5_1[0], batch_size)

        # backward student2
        # self.optimizer2.zero_grad()
        loss2 = sum([l.mean() for l in losses_dict_2.values()])
        loss2.backward()

        self.optimizer1.step()
        self.optimizer2.step()

        # collect student2 info
        acc1_2, acc5_2 = accuracy(preds_2, target, topk=(1, 5))
        train_meters_2["losses"].update(loss2.cpu().detach().numpy().mean(), batch_size)
        train_meters_2["top1"].update(acc1_2[0], batch_size)
        train_meters_2["top5"].update(acc5_2[0], batch_size)
        # print info
        msg1 = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters_1["data_time"].avg,
            train_meters_1["training_time"].avg,
            train_meters_1["losses"].avg,
            train_meters_1["top1"].avg,
            train_meters_1["top5"].avg,
        )
        msg2 = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters_2["data_time"].avg,
            train_meters_2["training_time"].avg,
            train_meters_2["losses"].avg,
            train_meters_2["top1"].avg,
            train_meters_2["top5"].avg,
        )
        return msg1, msg2




class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
