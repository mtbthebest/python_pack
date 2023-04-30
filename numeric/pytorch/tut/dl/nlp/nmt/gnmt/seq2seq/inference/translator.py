import time
import torch
from sacrebleu.metrics.bleu import BLEU
from seq2seq.data import config
import seq2seq.utils as utils
from seq2seq.inference.beam_search import SequenceGenerator


def gather_predictions(preds):
    return preds


def run_sacrebleu(test_path, reference_path):
    bleu = BLEU(tokenize='intl', effective_order=True)
    scores = []
    with open(test_path, 'r') as tfile, open(reference_path, 'r') as rfile:
        for tsent, rsent in zip(tfile, rfile):
            scores.append(bleu.sentence_score(tsent, [rsent]).score)

    return torch.mean(torch.tensor(scores))


class Translator:

    def __init__(self,
                 model,
                 tokenizer,
                 loader=None,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 print_freq=10,
                 reference=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.insert_tgt_start = [config.BOS]
        self.insert_src_start = [config.BOS]
        self.insert_src_end = [config.EOS]

        self.batch_first = model.batch_first
        self.beam_size = beam_size
        self.print_freq = print_freq
        self.reference = reference

        self.generator = SequenceGenerator(model=self.model,
                                           beam_size=beam_size,
                                           max_seq_len=max_seq_len,
                                           len_norm_factor=len_norm_factor,
                                           len_norm_const=len_norm_const,
                                           cov_penalty_factor=cov_penalty_factor)

    def run(self, calc_bleu=True, epoch=None, iteration=None, eval_path=None, summary=False,
            warmup=0, reference_path=None):
        if reference_path is None:
            reference_path = self.reference

        device = next(self.model.parameters()).device
        test_bleu = torch.tensor([0.], device=device)

        print(f'Running evaluation on test set {reference_path}')
        self.model.eval()

        output, eval_stats = self.evaluate(self.loader, epoch, iteration, warmup, summary)
        output = output[:len(self.loader.dataset)]
        output = self.loader.dataset.unsort(output)
        if eval_path:
            with open(eval_path, 'w') as eval_file:
                lines = [line + '\n' for line in output]
                eval_file.writelines(lines)
            if calc_bleu:
                test_bleu[0] = run_sacrebleu(eval_path, reference_path)
                print(f'BLEU on test dataset: {test_bleu[0]: 2f}')

        if calc_bleu:
            eval_stats['bleu'] = test_bleu[0].item()
        else:
            eval_stats['bleu'] = None

        print('Finished evaluation on test set')

        return output, eval_stats

    def evaluate(self, loader, epoch=0, iteration=0, warmup=0, summary=False):
        device = next(self.model.parameters()).device
        batch_time = utils.AverageMeter(warmup=warmup, keep=True)
        tot_tok_per_sec = utils.AverageMeter(warmup, keep=True)
        iterations = utils.AverageMeter()
        enc_seq_len = utils.AverageMeter()
        dec_seq_len = utils.AverageMeter()

        stats = {}

        batch_size = self.loader.batch_size
        beam_size = self.beam_size

        bos = [self.insert_tgt_start] * (batch_size * beam_size)
        bos = torch.tensor(bos, dtype=torch.int64, device=device)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        if beam_size == 1:
            generator = self.generator.greedy_search
        else:
            generator = self.generator.beam_search

        output = []

        for i, (src, indices) in enumerate(loader):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            translate_timer = time.time()
            src, src_length = src
            stats['total_enc_len'] = int(src_length.sum())
            src = src.to(device)
            src_length = src_length.to(device)
            with torch.no_grad():
                context = self.model.encode(src, src_length)
                context = [context, src_length, None]
                preds, lengths, counter = generator(batch_size, initial_input=bos, initial_context=context)

            stats['total_dec_len'] = lengths.sum().item()
            stats['iters'] = counter
            indices = torch.tensor(indices).to(preds)
            preds = preds.scatter(0, indices.unsqueeze(1).expand_as(preds), preds)
            preds = gather_predictions(preds).cpu()

            if self.tokenizer:
                for pred in preds:
                    pred = pred.tolist()
                    detok = self.tokenizer.detokenize(pred)
                    output.append(detok)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - translate_timer
            batch_time.update(elapsed, batch_size)

            total_tokens = stats['total_dec_len'] + stats['total_enc_len']
            ttps = total_tokens / elapsed
            tot_tok_per_sec.update(ttps, elapsed)

            iterations.update(stats['iters'])
            enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
            dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

            if i % self.print_freq == self.print_freq - 1:
                log = []
                log += 'TEST '
                if epoch is not None:
                    log += f'[{epoch}]'
                if iteration is not None:
                    log += f'[{iteration}]'

                log += f'[{i} / {len(loader)}]\t'
                log += f'Time {batch_time.val: 0.4f}({batch_time.avg: .4f})\t'
                log += f'Decoder iters {iterations.val: 0.1f}({iterations.avg: .1f})\t'
                log += f'Tok/s {tot_tok_per_sec.val: 0.1f}({tot_tok_per_sec.avg: .1f})\t'
                log = ''.join(log)
                print(log)

        eval_stats = {}
        eval_stats['tokens_per_sec'] = tot_tok_per_sec.avg
        eval_stats['runtimes'] = batch_time.avg
        eval_stats['throughouts'] = tot_tok_per_sec.vals

        return output, eval_stats
