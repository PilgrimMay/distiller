import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class MultiStudent(Distiller):
    def __init__(self, student, student2, teacher, cfg):
        super(MultiStudent, self).__init__(student,teacher)
        self.student2 = student2
        self.temperature = cfg.MS.TEMPERATURE
        self.ce_loss_weight = cfg.MS.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.MS.LOSS.KD_WEIGHT
        self.mutual_loss_weight = cfg.MS.LOSS.MUTUAL_WEIGHT

    def forward_train(self, image, target, epoch, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_student2, _ =self.student2(image)

        # logits_student2, _ =self.student2(image)

        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        if epoch > 150:
            loss_kd1 = (self.kd_loss_weight - self.mutual_loss_weight) * kd_loss(logits_student, logits_teacher,
                                                                                 self.temperature)
            loss_kd2 = (self.kd_loss_weight - self.mutual_loss_weight) * kd_loss(logits_student2, logits_teacher,
                                                                                 self.temperature)
        else:
            loss_kd1 = self.kd_loss_weight * kd_loss(logits_student, logits_teacher,
                                                                                 self.temperature)
            loss_kd2 = self.kd_loss_weight * kd_loss(logits_student2, logits_teacher,
                                                                                 self.temperature)

        # loss_kd1 = self.kd_loss_weight * kd_loss(logits_student, logits_teacher,
        #                                          self.temperature)
        # loss_kd2 = self.kd_loss_weight * kd_loss(logits_student2, logits_teacher,
        #                                          self.temperature)

        loss_mutual = 0
        if epoch > 150:
            print("mutual learning")
            if loss_kd1 > loss_kd2:
                # student2比student1掌握了更多teacher传递的知识, 因此student1应该向student2学习
                loss_mutual = kd_loss(logits_student, logits_student2, self.temperature)



        # loss_kd = loss_kd1

        loss_kd = loss_kd1 + loss_mutual

        # if loss_kd1 > loss_kd2:
        #     print("------------loss_kd1-----------:", loss_kd1)
        #     print("------------loss_kd-----------:", loss_kd)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }

        return logits_student, losses_dict




