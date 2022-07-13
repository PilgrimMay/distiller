import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

from mdistiller.engine.multi_train_utils import init_distributed_mode


# def main(cfg, resume, opts):
#     experiment_name = cfg.EXPERIMENT.NAME
#     if experiment_name == "":
#         experiment_name = cfg.EXPERIMENT.TAG
#     tags = cfg.EXPERIMENT.TAG.split(",")
#     if opts:
#         addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
#         tags += addtional_tags
#         experiment_name += ",".join(addtional_tags)
#     experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
#     if cfg.LOG.WANDB:
#         try:
#             import wandb
#
#             wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
#         except:
#             print(log_msg("Failed to use WANDB", "INFO"))
#             cfg.LOG.WANDB = False
#
#     # cfg & loggers
#     show_cfg(cfg)
#     # init dataloader & models
#     train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
#
#     # vanilla
#     if cfg.DISTILLER.TYPE == "NONE":
#         if cfg.DATASET.TYPE == "imagenet":
#             model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
#         else:
#             model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
#                 num_classes=num_classes
#             )
#         distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
#     # distillation
#     else:
#         print(log_msg("Loading teacher model", "INFO"))
#         if cfg.DATASET.TYPE == "imagenet":
#             model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
#             model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
#         else:
#             net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
#             assert (
#                 pretrain_model_path is not None
#             ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
#             model_teacher = net(num_classes=num_classes)
#             model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
#             model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
#                 num_classes=num_classes
#             )
#         if cfg.DISTILLER.TYPE == "CRD":
#             distiller = distiller_dict[cfg.DISTILLER.TYPE](
#                 model_student, model_teacher, cfg, num_data
#             )
#         else:
#             distiller = distiller_dict[cfg.DISTILLER.TYPE](
#                 model_student, model_teacher, cfg
#             )
#     distiller = torch.nn.DataParallel(distiller.cuda())
#
#     if cfg.DISTILLER.TYPE != "NONE":
#         print(
#             log_msg(
#                 "Extra parameters of {}: {}\033[0m".format(
#                     cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
#                 ),
#                 "INFO",
#             )
#         )
#
#     # train
#     trainer = trainer_dict[cfg.SOLVER.TRAINER](
#         experiment_name, distiller, train_loader, val_loader, cfg
#     )
#     trainer.train(resume=resume)

def main(cfg, resume, opts, args):

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # 初始化各进程环境 start
    init_distributed_mode(args)
    rank = args.rank
    world_size = args.world_size
    is_distributed = args.distributed
    # 初始化各进程环境 end


    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    if rank == 0:
        show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg, is_distributed)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        if rank == 0:
            print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            # model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
            #     num_classes=num_classes
            # )
            if cfg.DISTILLER.TYPE == "MultiStudent":
                model_student1 = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
                model_student2 = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            else:
                model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        elif cfg.DISTILLER.TYPE == "MultiStudent":
            distiller1 = distiller_dict[cfg.DISTILLER.TYPE](
                model_student1, model_student2, model_teacher, cfg
            )
            distiller2 = distiller_dict[cfg.DISTILLER.TYPE](
                model_student2, model_student1, model_teacher, cfg
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    if is_distributed == True:
        if cfg.DISTILLER.TYPE == "MultiStudent":
            distiller1 = torch.nn.parallel.DistributedDataParallel(distiller1.cuda(), device_ids=[args.gpu],
                                                                  find_unused_parameters=True)

            distiller2 = torch.nn.parallel.DistributedDataParallel(distiller2.cuda(), device_ids=[args.gpu],
                                                                  find_unused_parameters=True)

        else:
            distiller = torch.nn.parallel.DistributedDataParallel(distiller.cuda(), device_ids=[args.gpu],
                                                                  find_unused_parameters=True)

    else:
        if cfg.DISTILLER.TYPE == "MultiStudent":
            distiller1 = torch.nn.DataParallel(distiller1.cuda())
            distiller2 = torch.nn.DataParallel(distiller2.cuda())
        else:
            distiller = torch.nn.DataParallel(distiller.cuda())

    # if cfg.DISTILLER.TYPE == "MultiStudent":
    #     distiller1 = torch.nn.DataParallel(distiller1.cuda())
    #     distiller2 = torch.nn.DataParallel(distiller2.cuda())
    # else:
    #     distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        if cfg.DISTILLER.TYPE == "MultiStudent":
            print(
                log_msg(
                    "Extra parameters of {}: {}\033[0m".format(
                        cfg.DISTILLER.TYPE, distiller1.module.get_extra_parameters(),
                        cfg.DISTILLER.TYPE, distiller2.module.get_extra_parameters()
                    ),
                    "INFO",
                )
            )
        else:
            print(
                log_msg(
                    "Extra parameters of {}: {}\033[0m".format(
                        cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                    ),
                    "INFO",
                )
            )

    # train

    # print('-------------cfg.DISTILLER.TYPE----------',cfg.DISTILLER.TYPE)
    if cfg.DISTILLER.TYPE == "MultiStudent":
        trainer = trainer_dict[cfg.SOLVER.TRAINER](
            experiment_name, distiller1, distiller2, train_loader, val_loader, cfg
        )
        trainer.train(resume=resume)
    else:
        trainer = trainer_dict[cfg.SOLVER.TRAINER](
            experiment_name, distiller, train_loader, val_loader, cfg
        )
        trainer.train(rank, resume=resume)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts, args)
