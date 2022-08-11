"""
Prototypical Networks for Knowledge Disitllation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Focus(nn.Module):
    def __init__(self, c1=512, c2=256, k=1, s=1):
        super(Focus, self).__init__()
        self.pool = nn.AvgPool2d(8)
        self.conv = Conv(c1, c2, k, s)

    # bs x channels x h x w
    def forward(self, x):
        # print("x shape:", x.shape)
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], 1
        )
        x = self.conv(x)
        x = self.pool(x)
        avg = x.reshape(x.size(0), -1)

        return avg

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class PKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(PKD, self).__init__(student, teacher)
        self.temperature = cfg.PKD.TEMPERATURE
        self.ce_loss_weight = cfg.PKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.PKD.LOSS.KD_WEIGHT
        self.prototype_weight = cfg.PKD.LOSS.PROTOTYPE_WEIGHT
        # self.prototype = PrototypeNet()
        self.prototype = InformationPrototype(num_classes=100)

        self.feat_prototype_0 = FeaturePrototype(num_classes=100, channels=32,  H=32, W=32)
        self.feat_prototype_1 = FeaturePrototype(num_classes=100, channels=64,  H=32, W=32)
        self.feat_prototype_2 = FeaturePrototype(num_classes=100, channels=128, H=16, W=16)
        self.feat_prototype_3 = FeaturePrototype(num_classes=100, channels=256, H=8,  W=8)

        self.loss_func = cfg.PKD.LOSS_FUNC
        self.avgpool_s = nn.AvgPool2d(2)
        self.avgpool_t = nn.AvgPool2d(4)

        self.embed_s = Embed(256, 256)
        self.embed_t = Embed(256, 256)
        self.embed = Embed(160, 512)
        self.focus_s = Focus(512, 256, 1, 1)
        self.focus_t = Focus(512, 256, 1, 1)

    # include embedding
    def get_learnable_parameters(self):
        return super().get_learnable_parameters()
               # + list(self.embed_t.parameters())
               # list(self.embed_s.parameters()) + list(self.embed_t.parameters()) + \


    # without embedding
    # def get_learnable_parameters(self):
    #     return super().get_learnable_parameters() + list(self.prototype.parameters())

    def forward_train(self, image, target, **kwargs):
        '''

        # last year
        # res32x4 -> res32x8: 64 x 256 x 8 x 8
        # res56 -> res20: 64 x 64 x 8 x 8
        # res110 -> res32、res110 -> res20: 64 x 64 x 8 x 8
        # vgg13 -> vgg8：64 x 512 x 4 x 4
        # wrn_40_2 -> wrn_16_2: 64 x 128 x 8 x 8
        # wrn_40_2 -> wrn_40_1: 64 x 128 x 8 x 8 -> 64 x 64 x 8 x 8
        # ShuffleV1: 64 x 960 x 4 x 4
        # ShuffleV2: 64 x 464 x 4 x 4
        # vgg13->MobileNetV2: 64 x 512 x 4 x 4 -> 64 x 160 x 2 x 2

        # pooled_feat
        # res110 -> res32、res110 -> res20: 64 x 64
        # vgg13 -> vgg8：64 x 512
        # wrn_40_2 -> wrn_16_2: 64 x 128
        # res32x4 -> ShuffleV1: 64 x 256 -> 64 x 960
        # res32x4 -> ShuffleV2: 64 x 256 -> 64 x 1024
        # vgg13->MobileNetV2: 64 x 512 -> 64 x 1280
        # wrn_40_2 -> ShuffleV1: 64 x 128 -> 64 x 960
        # ResNet50 -> MobileNetV2: 64 x 2048  -> 64 x 1280
        # ResNet50 -> vgg8: 64 x 2048 -> 64 x 512

        :param image:
        :param target:
        :param kwargs:
        :return:
        '''
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # feat_s = self.focus_s(feature_student["feats"][3])
        # feat_t = self.focus_t(feature_teacher["feats"][3])

        # feat_3_stu = feature_student["feats"][3]
        # feat_3_tea = feature_teacher["feats"][3]

        # feat_2_stu = feature_student["feats"][2]
        # feat_2_tea = feature_teacher["feats"][2]

        # feat_1_stu = feature_student["feats"][1]
        # feat_1_tea = feature_teacher["feats"][1]

        # feat_0_stu = feature_student["feats"][0]
        # feat_0_tea = feature_teacher["feats"][0]

        #
        # last_feat_stu = feature_student["feats"][-1]
        # last_feat_tea = feature_teacher["feats"][-1]
        # last_feat_stu = self.avgpool_s(last_feat_stu)
        # last_feat_stu = self.embed(last_feat_tea.squeeze())
        # last_feat_tea = self.avgpool_t(last_feat_tea).squeeze()
        # feat_prototype_stu, feat_inter_class_stu = self.prototype(last_feat_stu, logits_student)
        # feat_prototypr_tea, feat_inter_class_tea = self.prototype(last_feat_tea, logits_teacher)
        # pkd_loss_feat = F.l1_loss(feat_prototype_stu, feat_prototypr_tea) + F.l1_loss(feat_inter_class_stu,
        #                                                                     feat_inter_class_tea)




        feat_stu = feature_student['pooled_feat']
        feat_tea = feature_teacher['pooled_feat']

        # feat_stu = self.embed_s(feat_stu)
        feat_tea = self.embed_t(feat_tea)

        # print("feat_stu shape:", feat_stu.shape)
        # print("feat_tea shape:", feat_tea.shape)


        # CE loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # KD loss
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )

        # prototype loss

        # feat_prototype_s_3, feat_inter_class_s_3 = self.feat_prototype_3(feat_3_stu, logits_student)
        # feat_prototype_t_3, feat_inter_class_t_3 = self.feat_prototype_3(feat_3_tea, logits_teacher)

        # feat_prototype_s_2, feat_inter_class_s_2 = self.feat_prototype_2(feat_2_stu, logits_student)
        # feat_prototype_t_2, feat_inter_class_t_2 = self.feat_prototype_2(feat_2_tea, logits_teacher)

        # feat_prototype_s_1, feat_inter_class_s_1 = self.feat_prototype_1(feat_1_stu, logits_student)
        # feat_prototype_t_1, feat_inter_class_t_1 = self.feat_prototype_1(feat_1_tea, logits_teacher)

        # feat_prototype_s_0, feat_inter_class_s_0 = self.feat_prototype_0(feat_0_stu, logits_student)
        # feat_prototype_t_0, feat_inter_class_t_0 = self.feat_prototype_0(feat_0_tea, logits_teacher)

        # pkd_loss_0 = F.l1_loss(feat_prototype_s_0, feat_prototype_t_0) + F.l1_loss(feat_inter_class_s_0,
        #                                                                            feat_inter_class_t_0)

        # pkd_loss_1 = F.l1_loss(feat_prototype_s_1, feat_prototype_t_1) + F.l1_loss(feat_inter_class_s_1,
        #                                                                            feat_inter_class_t_1)

        # pkd_loss_2 = F.l1_loss(feat_prototype_s_2, feat_prototype_t_2) + F.l1_loss(feat_inter_class_s_2,
        #                                                                            feat_inter_class_t_2)

        # pkd_loss_3 = F.l1_loss(feat_prototype_s_3, feat_prototype_t_3) + F.l1_loss(feat_inter_class_s_3,
        #                                                                            feat_inter_class_t_3)




        prototype_stu, inter_class_matrix_stu = self.prototype(feat_stu, logits_student)
        prototypr_tea, inter_class_matrix_tea = self.prototype(feat_tea, logits_teacher)

        pkd_loss_last = F.l1_loss(prototype_stu, prototypr_tea) + F.l1_loss(inter_class_matrix_stu,
                                                                                   inter_class_matrix_tea)

        # loss1 = self.prototype_weight * (F.l1_loss(prototype_stu, prototypr_tea) +
        #                                           F.l1_loss(inter_class_matrix_stu, inter_class_matrix_tea) +
        #                                           F.l1_loss(feat_prototype_s, feat_prototype_t) +
        #                                           F.l1_loss(feat_inter_class_s, feat_inter_class_t))
        #
        # loss2 = self.prototype_weight * (F.l1_loss(prototype_stu, prototypr_tea) +
        #                                  F.l1_loss(inter_class_matrix_stu, inter_class_matrix_tea))
        #
        # print(loss1>loss2)

        loss_prototype = self.prototype_weight * (pkd_loss_last)

        # loss_prototype = self.prototype_weight * (pkd_loss_last + pkd_loss_feat)


        # if self.loss_func == 'l1':
        #     loss_prototype = self.prototype_weight * (F.l1_loss(prototype_stu, prototypr_tea) +
        #                                               F.l1_loss(inter_class_matrix_stu, inter_class_matrix_tea))
        # elif self.loss_func == 'l2':
        #     loss_prototype = self.prototype_weight * (F.mse_loss(prototype_stu, prototypr_tea) +
        #                                               F.mse_loss(inter_class_matrix_stu, inter_class_matrix_tea))
        # else:
        #     raise NotImplementedError

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss prototype": loss_prototype
        }

        return logits_student, losses_dict

class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        # self.linear2 = nn.Linear(512, dim_out)
        self.l2norm = Normalize(2)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class FeaturePrototype(nn.Module):
    def __init__(self, num_classes, channels, H, W):
        super(FeaturePrototype, self).__init__()

        self.register_buffer('step', torch.tensor([0]*num_classes))
        self.register_buffer('prototypes', torch.zeros(num_classes, channels, H, W))

    def update_prototype(self, x_new, max_logits, max_cls, prototypes):
        '''

        :param x_new: 64 x 128 x 16 x 16
        :param max_logits: 64 x 1
        :param max_cls: 64 x 1
        :return: prototype: num_classes x channels x h x w  100 x 128 x 16 x 16
        '''


        num_classes = prototypes.shape[0]
        channels = prototypes.shape[1]
        H = prototypes.shape[2]
        W = prototypes.shape[3]
        local_ptypes = torch.zeros_like(prototypes)
        local_ptypes_cls_count = torch.zeros(num_classes).to(local_ptypes.device)
        for idx, cls in enumerate(list(max_cls)):
            local_ptypes[cls] += x_new[idx]
            local_ptypes_cls_count[cls] += 1

        exist_indx = local_ptypes_cls_count.type(torch.bool)
        local_ptypes[exist_indx] = local_ptypes[exist_indx] / (
                local_ptypes_cls_count[exist_indx].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1,channels, H, W)
            )
        prototypes = local_ptypes
        return prototypes

    def forward(self, x, class_logits):
        '''

        :param x: 64, 128, 16, 16
        :param class_logits: 64, 100
        :return:
        '''

        class_logits_new = class_logits.clone().detach()

        pred_normed = F.softmax(class_logits_new, dim=1)
        max_logit = pred_normed.max(dim=1)[0]  # 64
        max_cls = pred_normed.argmax(dim=1)  # 64

        x_new = x.clone().detach()


        self.prototypes = self.update_prototype(x_new, max_logit, max_cls, self.prototypes)

        num_classes = self.prototypes.shape[0]
        channels = self.prototypes.shape[1]
        p1 = self.prototypes.unsqueeze(0)
        p2 = torch.transpose(p1, 0, 1)
        inter_class_matrix = torch.zeros(num_classes, num_classes, channels)
        inter_class_matrix = p1 - p2

        prototypes = self.prototypes
        # intra_class_prototypes = self.intra_class_prototypes
        step = self.step

        return prototypes, inter_class_matrix



class InformationPrototype(nn.Module):
    def __init__(self, num_classes):
        super(InformationPrototype, self).__init__()

        self.register_buffer('step', torch.tensor([0]*num_classes))
        self.register_buffer('prototypes', torch.zeros(num_classes, 256))
        # self.inner = nn.Linear(2048, 2048)
        self.relu = nn.ReLU(inplace=True)


    def update_prototype(self, x_new, max_logits, max_cls, prototypes):
        '''

        :param x_new: 64 x 256
        :param max_logits: 64 x 1
        :param max_cls: 64 x 1
        :return: prototype: num_classes x channels 100 x 256
        '''

        num_classes, channels = prototypes.shape
        local_ptypes = torch.zeros_like(prototypes)
        local_ptypes_cls_count = torch.zeros(num_classes).to(local_ptypes.device)
        for idx, cls in enumerate(list(max_cls)):
            local_ptypes[cls] += x_new[idx]
            local_ptypes_cls_count[cls] += 1

        exist_indx = local_ptypes_cls_count.type(torch.bool)
        local_ptypes[exist_indx] = local_ptypes[exist_indx] / (
                local_ptypes_cls_count[exist_indx].unsqueeze(1).expand(-1,channels)
            )
        prototypes = local_ptypes
        return prototypes


    def graph(self, feat):
        '''

        :param feat: 64 x 256
        :return:
        '''
        N, D = feat.shape
        f1 = feat.unsqueeze(0).expand(N, N, D)
        f2 = feat.unsqueeze(1).expand(N, N, D)
        mat = F.cosine_similarity(f1, f2, dim=2)
        del f1, f2
        return mat


    def forward(self, x, class_logits):
        '''

        :param x: 64, 256, 8, 8
        :param class_logits: 64, 100
        :return:
        '''

        class_logits_new = class_logits.clone().detach()

        pred_normed = F.softmax(class_logits_new, dim=1)
        max_logit = pred_normed.max(dim=1)[0] # 64
        max_cls = pred_normed.argmax(dim=1) # 64

        x_mapped = x

        mat = self.graph(x_mapped)
        # x_aggregation = self.inner(torch.mm(mat, x_mapped).softmax(dim=-1)) + x_mapped
        # x_aggregation = torch.mm(mat, x_mapped).softmax(dim=-1) + x_mapped
        # x_aggregation = self.relu(torch.mm(mat, x_mapped)) + x_mapped
        x_aggregation = torch.mm(mat, x_mapped) + x_mapped

        x_new = x_aggregation.clone().detach()

        self.prototypes = self.update_prototype(x_new, max_logit, max_cls, self.prototypes)

        num_classes, channels = self.prototypes.shape
        p1 = self.prototypes.unsqueeze(0)
        p2 = torch.transpose(p1, 0, 1)
        inter_class_matrix = torch.zeros(num_classes, num_classes, channels)
        inter_class_matrix = p1-p2

        prototypes = self.prototypes
        step = self.step

        return prototypes, inter_class_matrix






