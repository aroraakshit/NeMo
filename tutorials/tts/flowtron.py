import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FlowtronModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="flowtron")
def main(cfg):
    
    trainer = pl.Trainer(**cfg.trainer)

    exp_manager(trainer, cfg.get("exp_manager", None))

    model = FlowtronModel(cfg=cfg.model, trainer=trainer)

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    trainer.fit(model)

if __name__ == '__main__':
    main()