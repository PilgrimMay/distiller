import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class SemCKD(Distiller):
    """SemCKD: Cross-Layer Distillation with Semantic Calibration"""

    def __init__(self, student, teacher, cfg):
        super(SemCKD, self).__init__(student, teacher)
        # self.student = student
        # self.teacher = teacher
        self.ce_loss_weight = cfg.SemCKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.SemCKD.LOSS.KD_WEIGHT
        self.fmd_loss_weight = cfg.SemCKD.LOSS.FMD_WEIGHT
        self.crit = nn.MSELoss(reduction='none')
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.num_stu_layer = len(self.student.get_stage_channels())
        self.num_tea_layer = len(self.teacher.get_stage_channels())
        self.stu_channels = self.student.get_stage_channels()[0:-1]
        self.tea_channels = self.teacher.get_stage_channels()[0:-1]
        self.attention_allocation = Attention_Allocation(self.num_stu_layer - 1, self.num_tea_layer - 1, self.batch_size,
                                              self.stu_channels, self.tea_channels)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.attention_allocation.parameters())

    def forward_train(self, image, target, **kwargs):
        # def __init__(self, stu_layer_len, tea_layer_len, input_channel, s_n, t_n, factor=4)
        # stu_layer_len, tea_layer_len, input_channel = batch size, s_n, t_n, factor = 4
        with torch.no_grad():
            # logits_teacher, feature_teacher = self.teacher(image)
            logits_teacher, feat_tea = self.teacher(image)
        feature_teacher = feat_tea['feats']

        # logits_student, feature_student = self.student(image)
        logits_student, feat_stu = self.student(image)
        feature_student = feat_stu['feats']

        # print("feature_student:",feature_student)

        # CE loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # KD loss
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, temperature=4)

        # SemCKD loss
        # num_stu_layer = len(feature_student['feats'])
        # num_tea_layer = len(feature_teacher['feats'])
        # num_stu_layer = len(feat_stu['feats'])
        # num_tea_layer = len(feat_tea['feats'])


        # print("---------num_stu_layer------:", num_stu_layer)
        # print("---------stu_layer----------:", len(self.student.get_stage_channels()))

        # batch_size = image.size(0)
        # input_channel = batch_size
        stu_channel = [f.shape[1] for f in feature_student[0:-1]]
        tea_channel = [f.shape[1] for f in feature_teacher[0:-1]]
        # if self.stu_channels == stu_channel:
        #     print("=====================True===================")

        # Attention allocation return: attention, proj_value_stu, value_tea

        # attention_allocation = Attention_Allocation(num_stu_layer - 2, num_tea_layer - 2, input_channel, stu_channel,
        #                                             tea_channel)
        # attention, proj_value_stu, value_tea = attention_allocation(feature_student[1:-1], feature_teacher[1:-1])
        attention, proj_value_stu, value_tea = self.attention_allocation(feature_student[0:-1], feature_teacher[0:-1])


        bsz, num_stu, num_tea = attention.shape
        assert self.batch_size == bsz, 'batch_size和bsz不等'

        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()
        for i in range(num_stu):
            for j in range(num_tea):
                # ind_loss[:, i, j] = self.crit(proj_value_stu[i][j], value_tea[i][j]).reshape(bsz,-1).mean(-1)
                ind_loss[:, i, j] = self.crit(proj_value_stu[i][j], value_tea[i][j]).reshape(bsz, -1).mean(-1)

        loss_SemCKD = (attention * ind_loss).sum() / (1.0 * bsz * num_stu)
        loss_SemCKD = self.fmd_loss_weight * loss_SemCKD

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_SemCKD": loss_SemCKD,
        }

        return logits_student, losses_dict


# class SemCKD(Distiller):
#     """SemCKD: Cross-Layer Distillation with Semantic Calibration"""
#
#     def __init__(self,student,teacher,cfg):
#         super(SemCKD, self).__init__(student,teacher)
#         # self.student = student
#         # self.teacher = teacher
#         self.ce_loss_weight = cfg.SemCKD.LOSS.CE_WEIGHT
#         self.kd_loss_weight = cfg.SemCKD.LOSS.KD_WEIGHT
#         self.fmd_loss_weight = cfg.SemCKD.LOSS.FMD_WEIGHT
#         self.crit = nn.MSELoss(reduction='none')
#
#
#     def forward_train(self, image, target, **kwargs):
#         # def __init__(self, stu_layer_len, tea_layer_len, input_channel, s_n, t_n, factor=4)
#         # stu_layer_len, tea_layer_len, input_channel = batch size, s_n, t_n, factor = 4
#         with torch.no_grad():
#             # logits_teacher, feature_teacher = self.teacher(image)
#             logits_teacher, feat_tea = self.teacher(image)
#         feature_teacher = feat_tea['feats']
#
#
#         # logits_student, feature_student = self.student(image)
#         logits_student, feat_stu = self.student(image)
#         feature_student = feat_stu['feats']
#
#
#         # print("feature_student:",feature_student)
#
#         # CE loss
#         loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
#
#         #KD loss
#         loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, temperature=4)
#
#         #SemCKD loss
#         # num_stu_layer = len(feature_student['feats'])
#         # num_tea_layer = len(feature_teacher['feats'])
#         num_stu_layer = len(feat_stu['feats'])
#         num_tea_layer = len(feat_tea['feats'])
#
#         # print("---------num_stu_layer------:", num_stu_layer)
#         # print("---------stu_layer----------:", len(self.student.get_stage_channels()))
#
#         batch_size = image.size(0)
#         input_channel = batch_size
#         stu_channel = [f.shape[1] for f in feature_student[1:-1]]
#         # s_n = [f.shape[1] for f in feat_s[1:-1]]
#         tea_channel = [f.shape[1] for f in feature_teacher[1:-1]]
#
#
#         # Attention allocation return: attention, proj_value_stu, value_tea
#
#         attention_allocation = Attention_Allocation(num_stu_layer-2, num_tea_layer-2, input_channel, stu_channel, tea_channel)
#         attention, proj_value_stu, value_tea = attention_allocation(feature_student[1:-1], feature_teacher[1:-1])
#         # print("proj_value_stu shape", proj_value_stu[0][0].shape)
#         # print("value_tea shape", value_tea[0][0].shape)
#
#
#
#         bsz, num_stu, num_tea = attention.shape
#         assert batch_size == bsz, 'batch_size和bsz不等'
#
#         ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()
#         for i in range(num_stu):
#             for j in range(num_tea):
#                 # ind_loss[:, i, j] = self.crit(proj_value_stu[i][j], value_tea[i][j]).reshape(bsz,-1).mean(-1)
#                 ind_loss[:, i, j] = self.crit(proj_value_stu[i][j], value_tea[i][j]).reshape(bsz, -1).mean(-1)
#
#         loss_SemCKD = (attention * ind_loss).sum()/(1.0*bsz*num_stu)
#         loss_SemCKD = self.fmd_loss_weight * loss_SemCKD
#
#         losses_dict = {
#             "loss_ce": loss_ce,
#             "loss_kd": loss_kd,
#             "loss_SemCKD": loss_SemCKD,
#         }
#
#         return logits_student, losses_dict




class Projection(nn.Module):
    """simple convolutional transformation"""
    def __init__(self,num_input_channels=1024,num_target_channels=128):
        super(Projection, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_target_channels = num_target_channels
        self.num_mid_channels = self.num_target_channels * 2

        def con1x1(in_channels,out_channels,stride=1):
            return nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=stride,bias=False)
        def con3x3(in_channels,out_channels,stride=1):
            return nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)

        self.regressor = nn.Sequential(
            con1x1(self.num_input_channels,self.num_mid_channels),
            nn.BatchNorm2d(self.num_mid_channels),
            nn.ReLU(inplace=True),
            con3x3(self.num_mid_channels,self.num_mid_channels),
            nn.BatchNorm2d(self.num_mid_channels),
            nn.ReLU(inplace=True),
            con1x1(self.num_mid_channels,self.num_target_channels),
        ).cuda()
        # self.regressor = self.regressor.cuda()

    def forward(self,x):
        x = self.regressor(x)
        return x

class Normalize(nn.Module):
    def __init__(self, power = 2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self,x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self,dim_in=1024,dim_out=128):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear1 = nn.Linear(dim_in, 2 * dim_out).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.linear2 = nn.Linear(2 * dim_out, dim_out).cuda()
        self.l2norm = Normalize(2).cuda()

    def forward(self,x):
        # print("---------------is_cuda---------------")
        # print("------dim_in------:",self.dim_in)
        # print("------dim_out------:", self.dim_out)
        # print("before x.view:", x.shape)

        x = x.view(x.shape[0], -1)
        # print("after x.view:", x.shape)
        # print("-------------------")
        # x = x.cpu()
        # x = self.l2norm(self.linear2(self.relu(self.linear1(x))))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.l2norm(self.linear2(x))
        # x = x.cuda()
        # print(x.is_cuda)
        return x


class Attention_Allocation(nn.Module):
    """Cross layer Self Attention"""
    def __init__(self, stu_layer_len, tea_layer_len, input_channel, s_n, t_n, factor = 4):
        super(Attention_Allocation, self).__init__()
        self.s_len = stu_layer_len
        self.t_len = tea_layer_len
        self.input_channel = input_channel

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for i in range(self.s_len):
            setattr(self, 'query_weight' + str(i), MLP(input_channel, input_channel // factor))
        for j in range(self.t_len):
            setattr(self, 'key_weight' + str(j), MLP(input_channel, input_channel // factor))

        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor' + str(i) + str(j), Projection(s_n[i], t_n[j]))


    def forward(self, feat_s, feat_t):

        similarity_maxtrix_stu = list(range(len(feat_s)))
        similarity_maxtrix_tea = list(range(len(feat_t)))
        bs = feat_s[0].shape[0]
        # bs = self.batch_size

        # print('--------------batch size---------------:', bs)

        for i in range(len(feat_s)):
            reshape_stu = feat_s[i].reshape(bs, -1)
            similarity_maxtrix_stu[i] = torch.matmul(reshape_stu, reshape_stu.t())
        for j in range(len(feat_t)):
            reshape_tea = feat_t[j].reshape(bs,-1)
            similarity_maxtrix_tea[j] = torch.matmul(reshape_tea, reshape_tea.t())

        # print("-------------similarity_maxtrix_stu---------------",similarity_maxtrix_stu[0].is_cuda)

        # print("-------------similarity_maxtrix_stu[0]------------",similarity_maxtrix_stu[0].shape)

        query = self.query_weight0(similarity_maxtrix_stu[0])
        query = query[:, None, :]

        for i in range(1, len(similarity_maxtrix_stu)):
            new_query = getattr(self, 'query_weight'+str(i))(similarity_maxtrix_stu[i])
            query = torch.cat([query, new_query[:, None, :]], 1)

        key = self.key_weight0(similarity_maxtrix_tea[0])
        key = key[:, :, None]
        for j in range(1, len(similarity_maxtrix_tea)):
            new_key = getattr(self, 'key_weight'+str(j))(similarity_maxtrix_tea[j])
            key = torch.cat([key, new_key[:, :, None]], 2)

        # attention
        queryXkey = torch.bmm(query, key)
        attention = F.softmax(queryXkey, dim = -1)

        # Projection
        proj_value_stu = []
        value_tea = []
        for i in range(len(similarity_maxtrix_stu)):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(len(similarity_maxtrix_tea)):
                stu_H = feat_s[i].shape[2]
                tea_H = feat_t[j].shape[2]
                if stu_H > tea_H:
                    # print('stu_H > tea_H')
                    input = F.adaptive_avg_pool2d(feat_s[i], (tea_H, tea_H))
                    proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(input))
                    value_tea[i].append(feat_t[j])
                elif stu_H < tea_H or stu_H == tea_H:
                    # print('stu_H < tea_H or stu_H == tea_H')
                    target = F.adaptive_avg_pool2d(feat_t[j], (stu_H, stu_H))
                    # print('feat_s[i] shape:',feat_s[i].shape)
                    # print('target shape:',target.shape)
                    proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(feat_s[i]))
                    value_tea[i].append(target)
                # print('            ')

        return attention, proj_value_stu, value_tea




