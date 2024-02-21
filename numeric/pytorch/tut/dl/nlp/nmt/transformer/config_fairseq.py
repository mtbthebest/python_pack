
class ConfigBase:
    ...




class SrcLangConfig:
    name = 'en'
    train_fname = f"train.tok.clean.dl.bpe.40000.{name}"
    valid_fname = f"valid.tok.clean.dl.bpe.40000.{name}"
    test_fname = f"test.tok.clean.dl.bpe.40000.{name}"
    max_seq_len = 128
    min_seq_len = 4


class TgtLangConfig:
    name = 'fr'
    train_fname = f"train.tok.clean.dl.bpe.40000.{name}"
    valid_fname = f"valid.tok.clean.dl.bpe.40000.{name}"
    test_fname = f"test.{name}"
    max_seq_len = 128
    min_seq_len = 4


class DatasetConfig(ConfigBase):
    src = SrcLangConfig
    tgt = TgtLangConfig
    path = f"/mnt/dl/Translation/WMT_15/{src.name}-{tgt.name}"
    vocab_fname = 'vocab.bpe.40000'
    bpe_fname = 'bpe.40000'
    size = None
    num_workers = 5


class TrainConfig(ConfigBase):
    dataset = DatasetConfig
    epochs = 10
    warmup_lr = 5e-6
    lr = 1e-4
    warmup_steps = 2000
    decay_factor = 0.4


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 0)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


class Config:
    
    
    @classmethod
    def _build(cls):
        base_architecture(cls)
        assert not hasattr(cls, 'device')
        assert not hasattr(cls, 'savepath')
        assert not hasattr(cls, 'train')
        assert not hasattr(cls, 'dataset')
        assert not hasattr(cls, 'batch_size')
        assert not hasattr(cls, 'seq_length')
        assert not hasattr(cls, 'total_batch_size')
        assert not hasattr(cls, 'grad_clip')
        cls.device = 'cuda'
        cls.savepath = '/mnt/dl/transformer_fairseq'
        cls.dataset = DatasetConfig
        cls.train = TrainConfig
        cls.batch_size = 64
        cls.seq_length = 128
        cls.total_batch_size = 16000
        cls.grad_clip = 5.
        return cls
    
    
    @classmethod
    def _asdict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and not callable(v)}
        

config = Config._build()
