
class ConfigBase:
    ...


class EncoderConfig(ConfigBase):
    vocab_size = 40000
    embed_dim = 512
    ffn_embed_dim = 2048
    layerdrop = 0.
    num_layers = 6
    num_heads = 8

    scale_embedding = True
    layernorm_embedding = True
    dropout_embedding = 0.1
    
    normalize_before = False
    
    dropout = 0.1
    activation_dropout = 0.1
    attention_dropout = 0.


class DecoderConfig(ConfigBase):
    vocab_size = 40000
    embed_dim = 512
    ffn_embed_dim = 2048
    layerdrop = 0.
    num_layers = 6
    num_heads = 8

    scale_embedding = True
    layernorm_embedding = True
    dropout_embedding = 0.1
    
    normalize_before = False
    
    
    dropout = 0.1
    activation_dropout = 0.1
    attention_dropout = 0.


class SrcLangConfig:
    name = 'en'
    train_fname = f"train.tok.clean.dl.bpe.40000.{name}"
    valid_fname = f"valid.tok.clean.dl.bpe.40000.{name}"
    test_fname = f"test.tok.clean.dl.bpe.40000.{name}"
    max_seq_len = 64
    min_seq_len = 4


class TgtLangConfig:
    name = 'fr'
    train_fname = f"train.tok.clean.dl.bpe.40000.{name}"
    valid_fname = f"valid.tok.clean.dl.bpe.40000.{name}"
    test_fname = f"test.{name}"
    max_seq_len = 64
    min_seq_len = 4


class DatasetConfig(ConfigBase):
    src = SrcLangConfig
    tgt = TgtLangConfig
    path = f"/mnt/dl/Translation/WMT_15/{src.name}-{tgt.name}"
    vocab_fname = 'vocab.bpe.40000'
    bpe_fname = 'bpe.40000'
    size = None
    num_workers = 2


class TrainConfig(ConfigBase):
    dataset = DatasetConfig
    epochs = 15
    # warmup_lr = 5e-6
    warmup_lr = 5e-7
    # warmup_lr = 1e-3
    # TODO 
    # lr = 1e-4
    lr = 3e-4
    # lr = 5e-3
    warmup_steps = 2000
    decay_factor = 0.4


class Config(ConfigBase):
    seed = 0
    padding_idx = 0
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    PAD = padding_idx
    UNK = 1
    BOS = 2
    EOS = 3

    # TODO 
    batch_size = 128
    seq_length = 128 * 100
    # TODO 
    # total_batch_size = 5120
    total_batch_size = 1280
    # total_batch_size = batch_size #16000
    # total_batch_size = batch_size#16000
    grad_clip = 1.

    train = TrainConfig
    # valid = ValidConfig

    encoder = EncoderConfig
    decoder = DecoderConfig
    dataset = DatasetConfig

    device = 'cuda'
    
    # TODO
    # savepath = '/mnt/dl/transformer2'
    savepath = '/mnt/dl/transformer_fp16'
