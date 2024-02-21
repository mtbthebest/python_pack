
import os
import numpy as np
import random
import math
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from model import TransformerModel
from dataset import LazyParallelDataset
from tokenizer import Tokenizer
from loss import SoftmaxScalerLoss
from config import Config
from lamb import LAMB
from scheduler import WarmupScheduler
from itertools import cycle
from utils import Embedding, Dropout, LayerNormalize, Linear
from positional_encoding import PositionalEmbedding, LearnedPositionalEmbedding
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

def get_epoch_seeds(generator, epochs):
    seeds = torch.randint(0, int(9e6), size=(epochs, ),
                          generator=generator).tolist()
    return seeds


def get_dataloader(cfg: Config, seeds):
    dataset_path = cfg.dataset.path
    train_src_fname = os.path.join(
        dataset_path, cfg.dataset.src.train_fname)
    train_tgt_fname = os.path.join(
        dataset_path, cfg.dataset.tgt.train_fname)
    train_max_len = cfg.dataset.src.max_seq_len
    train_min_len = cfg.dataset.src.min_seq_len

    tokenizer = Tokenizer(vocab_fname=os.path.join(dataset_path, cfg.train.dataset.vocab_fname, ),
                          bpe_fname=os.path.join(
        dataset_path, cfg.train.dataset.bpe_fname,),
        lang={'src': cfg.dataset.src.name,
              'tgt': cfg.dataset.tgt.name}
    )
    train_data = LazyParallelDataset(src_fname=train_src_fname,
                                     tgt_fname=train_tgt_fname,
                                     tokenizer=tokenizer,
                                     min_len=train_min_len,
                                     max_len=train_max_len,
                                     sort=False,
                                     max_size=cfg.dataset.size
                                     )
    train_loader = train_data.get_loader(batch_size=cfg.batch_size,
                                         seeds=seeds,
                                         batch_first=True,
                                         shuffle=True,
                                         batching="bucketing",
                                         batching_opt={
                                                  'num_buckets': 5},
                                         num_workers=cfg.train.dataset.num_workers,
                                         drop_last=True
                                         )
    cfg.encoder.vocab_size = tokenizer.vocab_size
    cfg.decoder.vocab_size = tokenizer.vocab_size
    return train_loader, tokenizer


class Trainer:

    def __init__(self, cfg: Config, train_loader, tokenizer, generator, seeds):
        self.train_loader = train_loader
        self.seeds = seeds
        self.generator = generator
        self.tokenizer = tokenizer

        self.model = TransformerModel(cfg)

        self.loss_fn = SoftmaxScalerLoss(reduction='sum', padding_idx=cfg.padding_idx)
        self.optimizer = LAMB(self.model.parameters(),
                              lr=cfg.train.lr,
                              weight_decay=1e-5)
        
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=cfg.train.lr, )

        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        
        steps_before_upd = cfg.total_batch_size // cfg.batch_size
        train_steps = (cfg.train.epochs *
                       len(self.train_loader) // steps_before_upd)
        print("Number of train steps for training is {}".format(train_steps))
        self.scheduler = WarmupScheduler(optimizer=self.optimizer,
                                         total_steps=train_steps,
                                         warmup_steps=cfg.train.warmup_steps,
                                         warmup_lr=cfg.train.warmup_lr,
                                         decay_factor=cfg.train.decay_factor
                                         )
        self.checkpoint_iter = iter(cycle(range(20)))
        self.savepath = cfg.savepath

        os.makedirs(self.savepath, exist_ok=True)

        self.total_batch_size = cfg.total_batch_size
        self.batch_size = cfg.batch_size
        self.epochs = cfg.train.epochs
        self.grad_clip = cfg.grad_clip

    def train(self, start_epoch=0):
        update_steps = self.total_batch_size // self.batch_size
        print("Global Train step is {}".format(update_steps))
        steps = 0
        self.model.train()
        self.lrs = []
        self.batch_losses = []
        self.optimizer.zero_grad()
        self.fp_scaler = GradScaler()
        grad_scale = 0.
        batch_data = {"src": [], "tgt": [], "dec_tgt": []}
        
        mini_steps = 0

        for i in range(start_epoch, self.epochs):
            print("Starting epoch {}...".format(i))
            self.train_loader.sampler.set_epoch(i)

            for j, (src, tgt, dec_tgt) in enumerate(self.train_loader):
                src, src_length = src
                tgt, tgt_length = tgt
                dec_tgt, dec_tgt_length = dec_tgt
                grad_scale += dec_tgt_length.sum().item()
                
                batch_data['src'].append(src)
                batch_data['tgt'].append(tgt)
                batch_data['dec_tgt'].append(dec_tgt)
                steps += 1
                
                
                if len(batch_data['src']) == update_steps:
                    # print("running update state ", grad_scale )
                    losses = []
                    for k in range(update_steps):
                        src = batch_data['src'][k].to(self.device)
                        tgt = batch_data['tgt'][k].to(self.device)
                        dec_tgt = batch_data['dec_tgt'][k].to(self.device)
                
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            out = self.model(src, tgt)
                            loss = self.loss_fn(out, dec_tgt, scale=grad_scale)
                        losses.append(loss.item() * grad_scale)
                        # print(loss, (loss * grad_scale))
                        self.fp_scaler.scale(loss).backward()
                    
                    self.batch_losses.append(np.sum(losses) / grad_scale)
                    self.fp_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.grad_clip
                                                  )
                    self.fp_scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.fp_scaler.update()
                    self.optimizer.zero_grad()
                    self.lrs.append(self.scheduler.get_last_lr()[0])
                    grad_scale = 0
                    batch_data['src'] = []
                    batch_data['tgt'] = []
                    batch_data['dec_tgt'] = []
                    print(
                        f'Epoch: {i}, Steps: {steps}, Batch: {j}/{len(self.train_loader)}, Loss: {self.batch_losses[-1] : 3.6f}, Lr: {self.scheduler.get_last_lr()[0]}')
            
                if steps % 5000 == 0:
                    next_check = next(self.checkpoint_iter)
                    self.save(os.path.join(self.savepath,
                                f"checkpoint{next_check:02}.pth"),
                                state=dict(epoch=i,
                                            losses=torch.tensor(
                                                self.batch_losses),
                                            lr=torch.tensor(self.lrs)
                                            ),

                                )
                    
            self.save(os.path.join(self.savepath, f"checkpoint-ep-{i:02}.pth"),
                      state=dict(epoch=i, losses=torch.tensor(self.batch_losses), lr=torch.tensor(self.lrs)))

    def save(self, pathname, state=None):
        if state is None:
            state = dict()
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": state.get("epoch", 0),
            "losses": state.get('losses', []),
            "lr": state.get('lr', []),
            "optimizer_name": self.optimizer.__class__.__name__
        }, pathname)

    
    def resume(self, ckp):
        state = torch.load(os.path.join(self.savepath, ckp))
        self.model.load_state_dict(state['model'])
        # self.optimizer.load_state_dict(state['optimizer'])
        # self.scheduler.load_state_dict(state['scheduler'])
        epoch = state['epoch']
        self.train(epoch + 1)
        

    def get_sample(self, sample_num=1):
        if sample_num > 10:
            raise NotImplementedError
        for i in range(0, 1):
            print("Getting sample...")
            ep = torch.randint(0, 10, (1, ))
            self.train_loader.sampler.set_epoch(ep.item())
            for j, (src, tgt, dec_tgt) in enumerate(self.train_loader):
                src, src_length = src
                tgt, tgt_length = tgt
                dec_tgt, dec_tgt_length = dec_tgt
                
                src = src[:sample_num]
                tgt = tgt[:sample_num]
                dec_tgt = dec_tgt[:sample_num]
                samples = dict()
                samples['tensors'] = {'src': src.clone(), 'tgt': tgt.clone(), 'dec_tgt': dec_tgt.clone()}
                samples['text'] = {'src': [], 'tgt': [], 'dec_tgt': []}
                for k in range(sample_num):
                    for key, tensor in zip(['src', 'tgt', 'dec_tgt' ], [src, tgt, dec_tgt]):
                        samples['text'][key].append([self.tokenizer.detokenize(tensor[k].numpy())])
                for key in ['src', 'tgt', 'dec_tgt' ]:
                    samples['text'][key] = np.concatenate(samples['text'][key])
                return samples
            

    def translate_random_sample(self):
        device = torch.device('cuda')
        is_training = self.model.training
        self.model.eval()
        self.model.cuda()
        size = len(self.train_loader.dataset)
        idx = torch.randint(0, size, (1, )).item()
        with torch.no_grad():
                sample = self.train_loader.dataset[idx]
                src = sample[0].to(device).unsqueeze(0)
                tgt = sample[1].to(device).unsqueeze(0)

                
                inps = tgt[:, :1]
                print('Src: ', self.tokenizer.detokenize(sample[0].numpy()))
                print('Tgt: ', self.tokenizer.detokenize(sample[1].numpy()))
                print("Starting translation... \n")
                for k in range(128):
                    out = self.model(src, inps)
                    probs = torch.softmax(out, dim=-1)
                    max_idx = torch.argmax(probs, dim=-1, )
                    inps = torch.column_stack((inps, max_idx[:, -1:]))
                    if inps[0, -1] == 3:
                        break
                print("Translation is: ", self.tokenizer.detokenize(inps.detach().cpu().numpy()[0]))

                return


