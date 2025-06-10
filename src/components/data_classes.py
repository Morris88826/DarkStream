from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class SpecArgs:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 160
    center: bool = False
    f_min: float = 0.0
    f_max: float = None
    streaming: bool = True


@dataclass
class SpecEncoderArgs:
    input_channels: int = 160
    depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    dims: list[int] = field(default_factory=lambda: [128, 256, 384, 512])
    drop_path_rate: float = 0.2
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    spec_args: SpecArgs = SpecArgs()
    lookahead: int = 0


@dataclass
class Lambdas:
    mel_loss: float = 15.0
    adv_feat_loss: float = 2.0
    adv_gen_loss: float = 1.0
    ped_loss: float = 20.0
    hubert_loss: float = 100000.0
    speaker_loss: float = 10.0
    commit_loss: float = 0.1


@dataclass
class CodecArgs:
    input_channels: int = 512
    output_channels: int = 512
    encode_channels: int = 512
    decode_channels: int = 512
    code_dim: int = 512
    codebook_num: int = 1
    codebook_size: int = 32
    bias: bool = True
    enc_ratios: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    dec_ratios: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    enc_strides: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    dec_strides: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    enc_kernel_size: int = 3
    dec_kernel_size: int = 3
    enc_block_dilations: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    enc_block_kernel_size: int = 3
    dec_block_dilations: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    dec_block_kernel_size: int = 3


@dataclass
class SeanetArgs:
    channels: int = 1
    dimension: int = 512
    causal: bool = True
    n_filters: int = 64
    n_residual_layers: int = 1
    activation: str = "ELU"
    compress: int = 2
    dilation_base: int = 2
    disable_norm_outer_blocks: int = 0
    kernel_size: int = 7
    residual_kernel_size: int = 3
    last_kernel_size: int = 3
    norm: str = "none"
    pad_mode: str = "constant"
    ratios: list[int] = field(default_factory=lambda: [8, 5, 4, 2])
    true_skip: bool = True
    look_ahead: bool = False
    look_ahead_size: int = 0


@dataclass
class SpecSeanetArgs:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = None
    dimension: int = 512
    n_residual_layers: int = 1
    layer_dims: list[int] = field(default_factory=lambda: [128, 256, 384, 512])
    activation: str = "ELU"
    norm: str = "none"
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_base: int = 2
    causal: bool = True
    pad_mode: str = "constant"
    true_skip: bool = True
    compress: int = 2
    disable_norm_outer_blocks: int = 0


@dataclass
class ConvNeXtArgs:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 160
    f_min: float = 0.0
    f_max: float = None
    input_channels: int = 160
    depths: list[int] = field(default_factory=lambda: [1, 1, 3, 1])
    dims: list[int] = field(default_factory=lambda: [128, 256, 384, 512])
    drop_path_rate: float = 0.2
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"
    look_ahead: bool = True
    look_ahead_size: int = 1


@dataclass
class TransformerArgs:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    causal: bool = True
    layer_scale: float = 0.01
    context: int = 250
    conv_layout: bool = True
    max_period: int = 10000
    gating: str = "none"
    norm: str = "layer_norm"
    positional_embedding: str = "rope"
    dim_feedforward: int = 2048
    input_dimension: int = 512
    output_dimensions: list[int] = field(default_factory=lambda: [512])


@dataclass
class VarianceAdapterArgs:
    in_channels: int = 512
    v_channels: int = 512
    out_channels: int = 512
    speaker_dim: int = 704
    var_kernel_size: int = 3
    pitch_kernel_size: int = 3
    energy_kernel_size: int = 3
    activation: str = "ELU"
    dropout: float = 0.3
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"


@dataclass
class DecSeanetArgs:
    channels: int = 1
    dimension: int = 512
    speaker_dim: int = 704
    causal: bool = True
    n_filters: int = 64
    n_residual_layers: int = 1
    activation: str = "ELU"
    compress: int = 2
    dilation_base: int = 2
    disable_norm_outer_blocks: int = 0
    kernel_size: int = 7
    residual_kernel_size: int = 3
    last_kernel_size: int = 3
    norm: str = "none"
    pad_mode: str = "constant"
    ratios: list[int] = field(default_factory=lambda: [8, 5, 4, 2])
    true_skip: bool = True

@dataclass
class Lambdas:
    mel_loss: float = 15.0
    adv_feat_loss: float = 2.0
    adv_gen_loss: float = 1.0
    ped_loss: float = 20.0
    hubert_loss: float = 100000.0
    speaker_loss: float = 10.0
    commitment_loss: float = 0.5


@dataclass
class ExpEncoderArgs:
    input_channels: int = 512
    depths: list[int] = field(default_factory=lambda: [1, 3, 1])
    dims: list[int] = field(default_factory=lambda: [512, 384, 512])
    drop_path_rate: float = 0.2
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"


@dataclass
class ProsodyEncoderArgs:
    input_size: int = 1024
    lin_neurons: int = 192
    activation: str = "ELU"
    channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    global_context: bool = True
    groups: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    dropout: float = 0.2

@dataclass
class ProsodyConditionerArgs:
    dim: int = 512
    dim_cond: int = 192


@dataclass
class DurationPredictorArgs:
    in_channels: int = 512
    out_channels: int = 512
    kernel_size: int = 3
    activation: str = "ELU"
    dropout: float = 0.3
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"


@dataclass
class QuantizerArgs:
    dimension: int = 512
    n_q: int = 1
    bins: int = 32
    input_dimension: int = 512
    output_dimension: int = 512


@dataclass
class ExpDecoderArgs:
    input_channels: int = 512
    depths: list[int] = field(default_factory=lambda: [2, 6, 2])
    dims: list[int] = field(default_factory=lambda: [256, 384, 512])
    drop_path_rate: float = 0.2
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"


@dataclass
class ConvPostnetArgs:
    in_channels: int= 512
    out_channels: int = 1024
    kernel_size: int = 7
    activation: str = "ELU"
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"
    use_postnet: bool = False


@dataclass
class ISTFTHeadArgs:
    dim: int = 512
    n_fft: int = 1024
    hop_length: int = 320


@dataclass
class VocosVocoderArgs:
    input_channels: int = 80
    dim: int = 512
    mlp_ratio: float = 3.0
    kernel_size: int = 7
    dilation: int = 1
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"
    num_layers: int = 8
    layer_scale_init_value: float = 1e-6


@dataclass
class VocoderFeatureArgs:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = None
    causal: bool = True
    pad_mode: str = "constant"