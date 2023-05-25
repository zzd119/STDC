import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import runner
from pytorch_lightning.utilities import rank_zero_info
from data.spatiotemprol import SpatioTemporalCSVDataModule
from datasets.PEMS04.generate_training_data import PEMS04_generate_data
from model.STDC.stdc_arch import STDC


def create_data(args):
    if os.path.exists(args.output_dir):
        if not args.cover:
            return
    else:
        os.makedirs(args.output_dir)
    if args.dataset_name == "PEMS04":
        PEMS04_generate_data(args)

def get_model(args):
    model = None
    if args.model_name == "STDC":
        model = STDC(**vars(args))
    return model

def get_task(args, model):
    task = getattr(runner, args.model_name + "Runner")(
        model=model, **vars(args)
    )
    return task

def get_data(args):
    dm = SpatioTemporalCSVDataModule(**vars(args))
    return dm

def runner_mian(args):

    rank_zero_info(vars(args))
    dm = get_data(args)
    model = get_model(args)
    task = get_task(args, model)
    trainer = pl.Trainer.from_argparse_args(args,num_nodes=1,callbacks=[ModelCheckpoint(monitor='Val_MAE',save_top_k=1,mode='min')])
    trainer.fit(task, dm)
    trainer.validate(datamodule=dm,ckpt_path='best')
    results = trainer.test(task, dm,ckpt_path='best')
    return results