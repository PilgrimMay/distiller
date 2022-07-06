from .trainer import BaseTrainer, CRDTrainer, MSTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "multistudent": MSTrainer,
}
