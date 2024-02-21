
import torch
from config import Config as cfg
# from trainer import get_epoch_seeds, Trainer, get_dataloader
from trainer_fp16 import get_epoch_seeds, Trainer, get_dataloader
# from trainer_scaler import get_epoch_seeds, Trainer, get_dataloader

def main():
    generator = torch.Generator()
    seeds = get_epoch_seeds(generator, cfg.train.epochs)
    train_loader, tokenizer = get_dataloader(cfg, seeds)


    trainer = Trainer(cfg=cfg, train_loader=train_loader, tokenizer=tokenizer,
                    generator=generator, seeds=seeds)
    
    trainer.train()
    # trainer.resume("/mnt/dl/transformer_lamb/checkpoint-ep-05.pth")
    

if __name__ == '__main__':
    main()