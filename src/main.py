import pyrallis
import copy
from configs.train_config import TrainConfig
from training.trainer import Trainer


@pyrallis.wrap()
def main(cfg: TrainConfig): 
    if cfg.log.inference:
        Trainer.inference(cfg, model_path=cfg.log.model_path, threshold=cfg.log.inference_threshold)
    else:
        cfg_copy = copy.deepcopy(cfg)
        trainer = Trainer(cfg)
        trainer.train()
        Trainer.inference(cfg_copy, model_path=cfg_copy.log.model_path, threshold=cfg_copy.log.inference_threshold)

if __name__ == '__main__':
    main()
