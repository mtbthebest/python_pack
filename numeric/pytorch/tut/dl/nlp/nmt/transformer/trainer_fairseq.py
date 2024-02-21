
import os
import numpy as np
import random
import math
import torch

from fairseq.models.transformer import TransformerModel
from dataset import LazyParallelDataset
from tokenizer import Tokenizer
from loss import SoftmaxLoss
from config_fairseq import config
from lamb import LAMB
from scheduler import WarmupScheduler
from itertools import cycle
from utils import Embedding, Dropout, LayerNormalize, Linear
from positional_encoding import PositionalEmbedding, LearnedPositionalEmbedding
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def get_epoch_seeds(generator, epochs):
    seeds = torch.randint(0, int(9e6), size=(epochs, ),
                          generator=generator).tolist()
    return seeds


def get_dataloader(seeds):
    dataset_path = config.dataset.path
    train_src_fname = os.path.join(
        dataset_path, config.dataset.src.train_fname)
    train_tgt_fname = os.path.join(
        dataset_path, config.dataset.tgt.train_fname)
    train_max_len = config.dataset.src.max_seq_len
    train_min_len = config.dataset.src.min_seq_len

    tokenizer = Tokenizer(vocab_fname=os.path.join(dataset_path, config.train.dataset.vocab_fname, ),
                          bpe_fname=os.path.join(
        dataset_path, config.train.dataset.bpe_fname,),
        lang={'src': config.dataset.src.name,
              'tgt': config.dataset.tgt.name}
    )
    train_data = LazyParallelDataset(src_fname=train_src_fname,
                                     tgt_fname=train_tgt_fname,
                                     tokenizer=tokenizer,
                                     min_len=train_min_len,
                                     max_len=train_max_len,
                                     sort=False,
                                     max_size=config.dataset.size
                                     )
    train_loader = train_data.get_loader(batch_size=config.batch_size,
                                         seeds=seeds,
                                         batch_first=True,
                                         shuffle=True,
                                         batching="bucketing",
                                         batching_opt={
                                                  'num_buckets': 5},
                                         num_workers=config.train.dataset.num_workers,
                                         drop_last=True
                                         )
    return train_loader, tokenizer


class Dictionary:

    def __init__(self, tokenizer) -> None:
        self.dictionary = tokenizer.tok2idx

    def __len__(self):
        return len(self.dictionary)

    def pad(self):
        return 0


class Task:
    def __init__(self, tokenizer) -> None:
        self.source_dictionary = Dictionary(tokenizer)
        self.target_dictionary = Dictionary(tokenizer)


class Model(torch.nn.Module):
    
    def __init__(self, tokenizer ) -> None:
        super().__init__()
        self.task = Task(tokenizer)
        self.net = TransformerModel.build_model(config, self.task)
        self.encoder = self.net.encoder
        self.decoder = self.net.decoder
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer = None,
        alignment_heads = None,
    ):

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out, _ = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


class Trainer:

    def __init__(self, train_loader, tokenizer, generator, seeds):
        self.train_loader = train_loader
        self.seeds = seeds
        self.generator = generator
        self.tokenizer = tokenizer
        
        self.model = Model(tokenizer)

        self.loss_fn = SoftmaxLoss(reduction='sum')
        # self.optimizer = LAMB(self.model.parameters(),
        #                       lr=config.train.lr,
        #                       weight_decay=1e-5)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.train.lr, )
        self.device = torch.device(config.device)
        self.model.to(self.device)

        steps_before_upd = config.total_batch_size // config.batch_size
        train_steps = (config.train.epochs *
                       len(self.train_loader) // steps_before_upd)
        print("Number of train steps for training is {}".format(train_steps))
        self.scheduler = WarmupScheduler(optimizer=self.optimizer,
                                         total_steps=train_steps,
                                         warmup_steps=config.train.warmup_steps,
                                         warmup_lr=config.train.warmup_lr,
                                         decay_factor=config.train.decay_factor
                                         )
        self.checkpoint_iter = iter(cycle(range(20)))
        self.savepath = config.savepath

        os.makedirs(self.savepath, exist_ok=True)
        

    def train(self, start_epoch=0):
        update_steps = config.total_batch_size // config.batch_size
        steps = 0
        # self.model_compiled.train()
        self.model.train()
        self.lrs = []
        losses = []
        self.batch_losses = []
        self.optimizer.zero_grad()
        grad_scale = 0.

        for i in range(start_epoch, config.train.epochs):
            print("Starting epoch {}...".format(i))
            self.train_loader.sampler.set_epoch(i)

            for j, (src, tgt, dec_tgt) in enumerate(self.train_loader):
                src, src_length = src
                tgt, tgt_length = tgt
                dec_tgt, dec_tgt_length = dec_tgt
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                dec_tgt = dec_tgt.to(self.device)
                # out = self.model_compiled(src, src_length, tgt)

                out = self.model(src, src_length, tgt)
                loss = self.loss_fn(out, dec_tgt)
                grad_scale += dec_tgt_length.sum().item()
                # print(loss)
                loss.backward()
                losses.append(loss.item())
                steps += 1
                if j== 1:
                    break
                self.scale_grad(grad_scale)
                
                # if steps and steps % update_steps == 0:
                self.update(grad_scale, losses)
                    # print(
                #     f'Epoch: {i}, Steps: {steps}, Batch: {j}/{len(self.train_loader)}, Loss: {self.batch_losses[-1] : 3.8f}, Lr: {self.scheduler.get_last_lr()[0]}')
                losses
                if steps and steps % (update_steps * 100) == 0:
                    next_check = next(self.checkpoint_iter)
                    self.save(os.path.join(self.savepath,
                                f"checkpoint{next_check:02}.pth"))
            
            self.save(os.path.join(self.savepath, f"checkpoint-ep-{i:02}.pth"))

    def save(self, pathname):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, pathname)

    def scale_grad(self, grad_scale):
        if grad_scale == 0.:
            raise Exception("division per zero")
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= grad_scale
    
    def generate(self, sample=0):
        device = torch.device('cpu')
        is_training = self.model.net.training
        self.model.net.eval()
        self.model.cpu()
        with torch.no_grad():
            self.train_loader.sampler.set_epoch(0)
            for j, (src, tgt, dec_tgt) in enumerate(self.train_loader):
                src, src_length = src
                tgt, tgt_length = tgt
                dec_tgt, dec_tgt_length = dec_tgt
                src = src.to(device)
                tgt = tgt.to(device)
                dec_tgt = dec_tgt.to(device)

                inps = tgt[sample:sample+1, :1]
                print(inps.shape, inps)
                print(self.tokenizer.detokenize(
                    src[sample].numpy()))
                print(self.tokenizer.detokenize(
                    tgt[sample].numpy()))
                src = src[sample:sample+1]
                src_length = src_length[sample:sample+1]
                print("Starting translation... \n")
                for k in range(128):
                    print(k)
                    # print(inps.shape, src.shape, src_length.shape)
                    out = self.model(src, src_length, inps)
                    # print(out.shape)
                    probs = torch.softmax(out, dim=-1)
                    max_idx = torch.argmax(probs, dim=-1, )
                    # print(max_idx[:, -1:])
                    inps = torch.column_stack((inps, max_idx[:, -1:]))
                    if inps[0, -1] == 3:
                        break
                    # print(inps)
                print(inps)
                
                print("Translation is: ", self.tokenizer.detokenize(inps.numpy()[0]))
                return

                  
        if is_training:
            self.model.train()
        self.model.cuda()

    def update(self, grad_scale, losses):
        self.scale_grad(grad_scale)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                        config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.batch_losses.append(
            np.array(losses).sum() / grad_scale)
        grad_scale = 0.
        self.lrs.append(self.scheduler.get_last_lr()[0])
        