
import os
import time
from itertools import cycle

import numpy as np
import torch

from seq2seq.train.optimizer import FP16Optimizer, FP32Optimizer
from seq2seq.train.scheduler import WarmupMultiStepLR
from seq2seq.utils import AverageMeter

class Seq2SeqTrainer:
    def __init__(self,
                 model,
                 criterion,
                 opt_config,
                 scheduler_config,
                 print_freq=10,
                 save_freq=1000,
                 save_dir=".",
                 save_info={},
                 grad_clip=float("inf"),
                 train_iterations=0,
                 keep_checkpoints=5,
                 checkpoint_filename='checkpoint%s.pth',
                 loss_scaling={},
                 iter_size=1,
                 intra_epoch_eval=0,
                 translator=None,
                 prealloc_mode="once",
                 warmup=1,
                 math="fp16",
                 verbose=True) -> None:

        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.device = next(model.parameters()).device
        self.print_freq = print_freq

        self.loss = None
        self.translator = translator
        self.intra_epoch_val = intra_epoch_eval
        self.warmup = warmup
        self.iter_size = iter_size

        self.prealloc_mode = prealloc_mode
        self.preallocated = False

        self.batch_first = model.batch_first
        self.verbose = verbose

        params = self.model.parameters()
        if math == "fp16":
            self.fp_optimzier = FP16Optimizer(model=model, grad_clip=grad_clip,
                                              loss_scale=loss_scaling['init_scale'],
                                              upscale=loss_scaling['upscale_interval'])
        elif math == "fp32":
            self.fp_optimzier = FP32Optimizer(model=model, grad_clip=grad_clip)

        optim_name = opt_config.pop("name")
        self.optimizer = self.initialize_optim(optim_name, params, **opt_config)

        self.scheduler = WarmupMultiStepLR(optimizer=self.optimizer,
                                           iterations=train_iterations,
                                           **scheduler_config)

    def initialize_optim(self, opt_name, params, **opt_config):
        if opt_name == 'Adam':
            optim = torch.optim.Adam(params, **opt_config)
        else:
            raise NotImplementedError

        return optim

    def load(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location={"cuda:0": 'cpu'})
        self.model.load_state_dict(checkpoint['state_dict'])
        self.fp_optimzier.initialize_model(self.model)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def optimize(self, dataloader):

        torch.set_grad_enabled(True)
        self.model.train()

        self.preallocate(dataloader.batch_size, dataloader.dataset.max_len, training=True)
        output = self.feed_data(dataloader, training=True)
        self.model.zero_grad()

        return output

    def preallocate(self, batch_size, max_length, training):
        if self.prealloc_mode == "always" or (self.prealloc_mode == 'once' and not self.preallocated):

            torch.cuda.empty_cache()

            src_length = torch.full((batch_size, ), max_length, dtype=torch.int64)
            tgt_length = torch.full((batch_size, ), max_length, dtype=torch.int64)

            if self.batch_first:
                shape = (batch_size, max_length)
            else:
                shape = (max_length, batch_size)

            src = torch.full(shape, 4, dtype=torch.int64)
            tgt = torch.full(shape, 4, dtype=torch.int64)

            src = src, src_length
            tgt = tgt, tgt_length

            self.iterate(src, tgt, update=False, training=training)

            self.model.zero_grad()
            self.preallocated = True

    def iterate(self, src, tgt, update=True, training=True):
        src, src_length = src
        tgt, tgt_length = tgt
        # print("received src", src_length[:3], src[:3, :3])
        # print("received tgt", tgt_length[:3])

        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            seq_size, batch_size = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            seq_size, batch_size = output.size(0), output.size(1)

        loss = self.criterion(output.view(batch_size * seq_size, -1),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()

        loss /= (batch_size * self.iter_size)
        # print(self.criterion)
        # print("My loss ",loss)
        # raise Exception

        if training:
            self.fp_optimzier.step(loss, self.optimizer, self.scheduler, update)
        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / batch_size
        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, dataloader, training=True):
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_val + 2)[1:-1]
            iters_with_update = len(dataloader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)
        batch_time = AverageMeter(self.warmup)
        data_time = AverageMeter(self.warmup)
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tok_tok_time = AverageMeter(self.warmup)
        src_tok_time = AverageMeter(self.warmup)
        tgt_tok_time = AverageMeter(self.warmup)

        batch_size = dataloader.batch_size

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        for i, (src, tgt) in enumerate(dataloader):
            self.save_counter += 1
            data_time.update(time.time() - end)

            update = False
            if i % self.iter_size == self.iter_size - 1:
                update = True

            stats = self.iterate(src, tgt, update, training=training)

            loss_per_token, loss_per_sentence, num_toks = stats

            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed, elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed, elapsed)
            tok_tok_time.update((num_toks['src'] + num_toks['tgt']) / elapsed, elapsed)
            self.loss = losses_per_token.avg

            if training and i in eval_iters:
                eval_fname = f"eval_epoch_{self.epoch}_iter{i}"
                eval_path = os.path.join(self.save_dir, eval_fname)
                _, eval_stats = self.translator.run(calc_bleu=True,
                                                    epoch=self.epoch,
                                                    iteration=i,
                                                    eval_path=eval_path)

                test_bleu = eval_stats['bleu']

                log = []
                log += [f"TRAIN [{self.epoch}][{i} / {len(dataloader)}]"]
                log += [f'BLEU: {test_bleu: .2f}']
                log = "\t".join(log)

                print(log)

                self.model.train()

                self.preallocate(dataloader.batch_size,
                                 dataloader.dataset.max_len,
                                 training=True)

            if i % self.print_freq == 0:
                phase = "TRAIN" if training else "VALIDATION"
                log = []
                log += [f"{phase}[{self.epoch}][{i} / {len(dataloader)}]"]
                log += [f"Time {batch_time.val: .3f}({batch_time.avg: .3f})"]
                log += [f"Data {data_time.val: .3f}({data_time.avg: .3f})"]
                log += [f"Tok/s {tok_tok_time.val: .3f}({tok_tok_time.avg: .3f})"]

                if self.verbose:
                    log += [f"Src tok/s {src_tok_time.val: .0f} ({src_tok_time.avg})"]
                    log += [f"Tgt tok/s {tgt_tok_time.val: .0f} ({tgt_tok_time.avg})"]
                    log += [f'Loss/sentence: {losses_per_sentence.val: .6f} ({losses_per_sentence.avg: .6f})']
                log += [f'Loss/tok: {losses_per_token.val: .6f} ({losses_per_token.avg: .6f})']

                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    log += [f"LR {lr: .3e}"]
                log = '\t'.join(log)
                print(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    self.save(identifier=identifier)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

        return losses_per_token.avg, tok_tok_time.avg

    def save(self, identifier, is_best=False, save_all=False):

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_dir, filename)
            print(f"Saving model to {filename}")
            torch.save(state, filename)

        model_state = self.model.state_dict()

        state = {
            "epoch": self.epoch,
            "state_dict": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss": getattr(self, 'loss', None)
        }

        state = dict(list(state.items()) + list(self.save_info.items()))
        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = "model_best.pth"
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)

    def evaluate(self, dataloader):
        torch.set_grad_enabled(False)
        self.model.eval()

        self.preallocate(dataloader.batch_size, dataloader.dataset.max_len, training=False)
        output = self.feed_data(dataloader, training=False)

        self.model.zero_grad()

        return output
