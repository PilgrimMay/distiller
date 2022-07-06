import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    loss_kd *= temperature**2
    return loss_kd

class TaT(Distiller):
    def __init__(self, student, teacher, cfg):
        super(TaT, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.TAT.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.TAT.LOSS.KD_WEIGHT
        self.tat_loss_weight = cfg.TAT.LOSS.TAT_WEIGHT
        self.temperature = cfg.TAT.TEMPERATURE
        self.patch_n = cfg.TAT.PATCH_N
        self.patch_m = cfg.TAT.PATCH_N
        self.group_num = cfg.TAT.GROUP
        # self.batch_transformation = batch_transformation()
        self.transformation = transformation()
        self.bs = cfg.SOLVER.BATCH_SIZE

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.transformation.parameters())

    def forward_train(self, image, target, **kwargs):
        logits_student, feat_stu = self.student(image)
        with torch.no_grad():
            logits_teacher, feat_tea = self.teacher(image)

        feature_student = feat_stu["feats"][3]
        feature_teacher = feat_tea["feats"][3]
        # print("----------feature_student----------", feature_student.shape)




        stu_patch_group = patch_group(feature_student)
        tea_patch_group = patch_group(feature_teacher)

        #TAT loss

        tat_list = []
        # loss_tat = sum([F.mse_loss(self.transformation(stu_patch, tea_patch) for )])
        # len = len(stu_patch_group)
        for i in range(len(stu_patch_group)):
            stu_patch = stu_patch_group[i]
            # print("--------------stu_patch shape-------------", stu_patch.shape)
            tea_patch = tea_patch_group[i]

            # f_s = torch.flatten(self.regressor(feat_stu), start_dim=2)
            t_target = torch.flatten(tea_patch,start_dim=2)
            t_target = torch.transpose(t_target, 1, 2)

            # print("---------teacher patch----------:",tea_patch.shape)

            # s_n = stu_patch.shape[i]
            # t_n = tea_patch_group[i]

            s_target = self.transformation(stu_patch, tea_patch)

            tat_list.append(F.mse_loss(s_target,t_target))

            # loss_tat += F.mse_loss(s_target, tea_patch)

            # for j in range(self.bs):
            #     s = stu_patch[j, :, :, :]
            #     t = tea_patch[j, :, :, :]
            #     self_target = self.batch_transformation(s, t)
            #     loss_tat += F.mse_loss(s, t)
        loss_tat = sum(tat_list)



        # CE loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # KD loss
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)



        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_tat": loss_tat,
        }

        return logits_student, losses_dict

# def single_stage_at_loss(stu)


def patch_group(feat_map, n=2, m=2, group=2):
    H = feat_map.shape[2]
    W = feat_map.shape[3]
    assert H % n == 0, "H无法整除n，不能进行patch group操作"
    assert W % m == 0, "W无法整除m，不能进行patch group操作"
    h = H // n
    w = W // m
    assert n * m % group == 0, "n*m无法整除group, 不能进行patch group操作"
    patches_per_group = n * m // group


    patch_list = []
    for i in range(n):
        for j in range(m):
            patch_list.append(feat_map[:, :, i*h:(i+1)*h, j*w:(j+1)*w])

    patch_group = []
    # temp_list = []
    for i in range(group):
        # patch_group.append(torch.cat())
        temp_list = patch_list[i*patches_per_group:(i+1)*patches_per_group]
        temp_patch = temp_list[0]
        for j in range(len(temp_list)):
            if j == 0:
                continue
            temp_patch = torch.cat([temp_patch, temp_list[j]], dim=1)
            if j == len(temp_list)-1:
                patch_group.append(temp_patch)
        temp_list.clear()

    return patch_group


class Conv(nn.Module):
    def __init__(self, num_input_channels=512, num_target_channels=512):
        super(Conv, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_target_channels = num_target_channels

        def con3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            con3x3(self.num_input_channels,self.num_target_channels),
            nn.BatchNorm2d(self.num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class transformation(nn.Module):
    def __init__(self, num_input_channels=512, num_target_channels=512):
        super(transformation, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_target_channels = num_target_channels
        self.regressor = Conv(self.num_input_channels, self.num_target_channels)

    def forward(self, feat_stu, feat_tea):
        f_s = torch.flatten(self.regressor(feat_stu), start_dim=2) # B x C x N

        f_t = torch.flatten(self.regressor(feat_tea), start_dim=2) # B x C x N
        f_self = torch.flatten(self.regressor(feat_stu), start_dim=2) # B x C x N
        f_s = torch.transpose(f_s, 1, 2) # B x N x C
        f_self = torch.transpose(f_self, 1, 2) # B x N xC
        # f_t = torch.transpose(f_t, 1, 2) # B x N x C
        # print("---------f_self shape---------:", f_s.shape[1])
        sxt = torch.bmm(f_s, f_t)
        # print("---------sxt shape------------:", sxt.shape)

        w = F.softmax(sxt, dim=1)
        # print("----------weight shape---------:", w.shape[2])

        s_target = torch.bmm(w, f_self)

        # print("----------s_target shape---------:", s_target.shape)

        return s_target




# class transformation(nn.Module):
    # def __init__(self, num_input_channels, num_target_channels):
    #     super(transformation, self).__init__()
    #
    # def forward(self, feat_stu, feat_tea, num_input_channels, num_target_channels): # feature shape: C x H x W
    #     # f_s = self.regressor(feat_stu)
    #     # f_t = self.regressor(feat_tea)
    #     # f_self = self.regressor(feat_stu)
    #     self.regressor = Conv(num_input_channels, num_target_channels)
    #
    #     f_s = torch.flatten(self.regressor(feat_stu), start_dim=1)  # shape: C x N, N = H x W
    #     f_t = torch.flatten(self.regressor(feat_tea), start_dim=1)
    #     f_self = torch.flatten(self.regressor(feat_stu), start_dim=1)
    #     s_target = torch.zeros_like(f_s)
    #
    #     C = f_s.shape[1]
    #     N = f_s.shape[2]
    #
    #     weight = torch.zeros(N)
    #
    #     for i in range(N):
    #         f_t_i = f_t[: ,i]  #  C x 1
    #         for j in range(N):
    #             f_s_j = f_s[:, j] #  C x 1
    #             weight.append(torch.dot(f_t_i,f_s_j))
    #         norm = weight.sum()
    #         normlize = weight.div(norm)
    #         for k in range(N):
    #             s_target[i] += normlize[k] * f_self[:, k]
    #
    #     return s_target

# class batch_transformation(nn.Module):
#     def __init__(self):
#         super(batch_transformation, self).__init__()
#
#     def forward(self, feature_student, feature_teacher, num_input_channels, num_target_channels):
#         bs = feature_student.shape[0]
#
#         f_s = torch.zeros_like(feature_student)
#         self.transformation = transformation(num_input_channels, num_target_channels)
#         for i in range(bs):
#             feat_stu = feature_student[i]
#             feat_tea = feature_teacher[i]
#             f_s[i] = self.transformation(feat_stu, feat_tea)
#
#         return f_s



