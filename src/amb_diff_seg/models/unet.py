import math
import torch

import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from torch import nn
from omegaconf import OmegaConf

from models.nn import (SiLU,conv_nd,linear,avg_pool_nd,zero_module,normalization,
                              timestep_embedding,checkpoint,identity_module,total_model_norm)
from models.fp16 import (convert_module_to_f16,
                                         convert_module_to_f32)
from models.gen_prob_unet import GenProbUNet

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention__init__
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        image_size=32,
        out_channels=1,
        in_channels=4,
        model_channels=64,
        num_res_blocks=3,
        num_middle_res_blocks=2,
        attention_resolutions="-2,-1",
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        no_diffusion=False,
        final_act="none",
        one_skip_per_reso=False,
        new_upsample_method=False,
        mlp_attn=False
    ):
        super().__init__()
        self.mlp_attn = mlp_attn
        self.new_upsample_method = new_upsample_method
        self.one_skip_per_reso = one_skip_per_reso
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        time_embed_dim = model_channels*4
        if isinstance(num_res_blocks,int):
            num_res_blocks = [num_res_blocks]*len(channel_mult)
        assert len(num_res_blocks) == len(channel_mult), f"len(num_res_blocks): {len(num_res_blocks)} must be equal to len(channel_mult): {len(channel_mult)}"

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        self.attention_resolutions = []
        for ar in attention_resolutions:
            if ar < 0:
                ar = len(channel_mult) + ar
            self.attention_resolutions.append(ar)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample

        self.class_dict = {}

        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.no_diffusion = no_diffusion
        self.fp16_attrs = ["input_blocks","output_blocks"]
        if num_middle_res_blocks>=1:
            self.fp16_attrs.append("middle_block")

        self.fp16_attrs.append("time_embed")
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.in_channels = in_channels

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, self.in_channels, model_channels, 3, padding=1))
            ])
        self.input_skip = [False]
        input_block_chans = [model_channels]
        ch = model_channels
        res_block_kwargs = {"emb_channels": time_embed_dim,
                            "dropout": dropout,
                            "dims": dims,
                            "use_checkpoint": use_checkpoint,
                            "use_scale_shift_norm": use_scale_shift_norm}
        attn_kwargs = {"use_checkpoint": use_checkpoint,
                        "num_heads": num_heads,
                        "with_xattn": False,
                        "xattn_channels": None}
        resolution = 0
        
        assert channel_mult[0]==1, "channel_mult[0] must be 1"
        for level, (mult, n_res_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            for _ in range(n_res_blocks):
                if self.new_upsample_method:
                    ch = mult*model_channels
                    ch_in = ch
                else:
                    ch_in = ch
                    ch = mult*model_channels
                layers = [
                    
                ]
                
                if resolution in self.attention_resolutions:
                    if self.mlp_attn:
                        layers = [MLPBlock(ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                    else:
                        layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_skip.append(False)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                resolution += 1
                ch_out = channel_mult[resolution]*model_channels if self.new_upsample_method else None
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, channels_out=ch_out))
                )
                self.input_skip[-1] = True
                self.input_skip.append(False)
                input_block_chans.append(ch)
        if resolution in self.attention_resolutions:
            if self.mlp_attn:
                middle_layers = (sum([[MLPBlock(ch,**res_block_kwargs),
                               AttentionBlock(ch,**attn_kwargs)] 
                               for _ in range(num_middle_res_blocks-1)],[])+
                            [MLPBlock(ch,**res_block_kwargs)])
            else:
                middle_layers = (sum([[ResBlock(ch,**res_block_kwargs),
                               AttentionBlock(ch,**attn_kwargs)] 
                               for _ in range(num_middle_res_blocks-1)],[])+
                            [ResBlock(ch,**res_block_kwargs)])
        else:
            middle_layers = [ResBlock(ch,**res_block_kwargs) for _ in range(num_middle_res_blocks)]

        self.middle_block = TimestepEmbedSequential(*middle_layers)

        attn_kwargs["num_heads"] = num_heads_upsample
        self.output_blocks = nn.ModuleList([])
        for level, mult, n_res_blocks in zip(reversed(list(range(len(channel_mult)))),channel_mult[::-1],num_res_blocks[::-1]):
            for i in range(n_res_blocks + 1):
                if self.new_upsample_method:
                    ch = model_channels * mult
                    ch_in = ch
                else:
                    ch_in = ch+input_block_chans.pop()
                    ch = model_channels * mult
                if resolution in self.attention_resolutions:
                    if self.mlp_attn:
                        layers = [MLPBlock(ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                    else:
                        layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                                  AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                if level and i == n_res_blocks:
                    resolution -= 1
                    ch_out = channel_mult[resolution]*model_channels if self.new_upsample_method else None
                    layers.append(Upsample(ch, conv_resample, dims=dims, channels_out=ch_out,
                                           mode="bilinear" if self.new_upsample_method else "nearest"))
                    
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        if self.one_skip_per_reso:
            assert self.new_upsample_method, "one_skip_per_reso only works with new_upsample_method"
        else:
            self.input_skip = [True for _ in self.input_skip]

        final_act_dict = {"none": nn.Identity(),
                       "softmax": nn.Softmax(dim=1),
                          "tanh": nn.Tanh()}
        self.out = nn.Sequential(
            nn.Identity(),#unnecessary, but kept for key consistency
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
            final_act_dict[final_act.lower()]
        )
        self.out_channels = out_channels

    def initialize_as_identity(self,verbose=False):
        """Initializes parameters in all modules such that the model behaves as an identity function. 
        Convolutions are initialized with zeros, except for a 1 in their central pixel. Bias terms are set to zero.
        BatchNorm layers are initialized such that their output is zero. 
        """
        start_names = ['input_blocks', 'middle_block', 'output_blocks', 'out']
        success_params = 0
        total_params = 0
        t_before = total_model_norm(self)
        total_params = sum([p.numel() for p in self.parameters()])
        for name,m in self.named_modules():            
            if isinstance(m,(nn.Linear,nn.Conv2d)) and any([name.startswith(n) for n in start_names]):
                success = identity_module(m,raise_error=False)
                if success:
                    success_params += sum([p.numel() for p in m.parameters()])
        t_after = total_model_norm(self)
        if verbose:
            print(f"Initialized {success_params}/{total_params} ({100*success_params/total_params:.2f}%) parameters as identity")
            print(f"Model norm before: {t_before:.2f}, after: {t_after:.2f}, relative change: {100*(t_after-t_before)/t_before:.2f}%")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def to_xattn(self, vit_output, depth):
        if self.vit_injection_type!="xattn":
            out = None
        else:
            if self.block_info.iloc[depth]["has_attention"]:
                out = vit_output
            else:
                out = None
        return out

    def apply_class_emb(self, classes):
        emb = 0
        for i,k in enumerate(self.class_dict.keys()):
            emb += self.class_emb[k](classes[:,i])
        return emb

    def forward(self, image, sample, timesteps):
        """
        Apply the model to an input batch.

        :param sample: an [N x C x ...] Diffusion sample tensor.
        :param timesteps: a 1-D batch of timesteps or a 0-D single timestep to repeat.
        :param image: an [N x C x ...] image tensor.
        :return: an [N x C x ...] Tensor of predicted masks.
        """
        bs = sample.shape[0]
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(bs)
        assert image.shape[1]==3 and sample.shape[1]==1, f"image shape: {image.shape}, sample shape: {sample.shape}" #TODO remove
        h = torch.cat([sample, image], dim=1).type(self.inner_dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        depth = 0
        for module,skip in zip(self.input_blocks,self.input_skip):
            h = module(h, emb)
            if skip:
                hs.append(h)
            else:
                hs.append(0)
            depth += 1
        h = self.middle_block(h, emb)
        depth += 1
        for module in self.output_blocks:
            if self.new_upsample_method:
                cat_in = h + hs.pop()
            else:
                cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            depth += 1
        h = h.type(sample.dtype)
        h = self.out(h)
        return h

def create_unet_from_args(args):
    if args["final_act"]=="tanh_if_x":
        if args["predict"]=="x":
            final_act = "tanh"
        else:
            final_act = "none"
    else:
        final_act = args["final_act"]

    if args["use_gen_prob_unet"]:
        return GenProbUNet(final_act=final_act)
    image_size = 64 if args["random_crop64"] else 128
    unet = UNetModel(image_size=image_size, #128
                num_res_blocks=args["num_res_blocks"], #[1,2,3,4]
                model_channels=args["num_channels"],#32
                attention_resolutions=args["attention_resolutions"],#-1
                dropout=args["dropout"],#0.0
                channel_mult=args["channel_mult"],#(1,1,2,4)
                num_heads=args["num_heads"],#4
                num_heads_upsample=args["num_heads_upsample"],#-1
                final_act=final_act,#"tanh"
                one_skip_per_reso=args["one_skip_per_reso"],#False
                new_upsample_method=args["new_upsample_method"],#False
                mlp_attn=args["mlp_attn"],#False
                num_middle_res_blocks=args["num_middle_res_blocks"],#4
                )
    if args["identity_init"]:
        unet.initialize_as_identity()
    return unet



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x,y=x_attn)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class MLPBlock(TimestepBlock):
    """
    Based on the MLP block from SiD (simple diffusion) pseudo code.

    def mlp_block(x, emb, expansion_factor=4):
    B, HW, C = x.shape
    x = Normalize(x)
    mlp_h = Dense(x, expansion_factor * C)
    scale = DenseGeneral(emb, mlp_h.shape [2:])
    shift = DenseGeneral(emb, mlp_h.shape [2:])
    mlp_h = swish(mlp_h)
    mlp_h = mlp_h * (1. + scale [:, None ]) + shift [:, None]
    if config.transformer_dropout > 0.:
        mlp_h = Dropout(mlp_h, config.transformer_dropout)
    out = Dense(mlp_h, C, kernel_init = zeros)
    return out"""
    def __init__(
        self,
        channels,
        emb_channels,
        expansion_factor=4,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        c = expansion_factor * channels
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            normalization(channels),
            conv_nd(dims, channels, c, 1),
            SiLU(),
        )
        self.emb_layers = linear(
                emb_channels,
                2 * c if use_scale_shift_norm else c,
            )
        self.out_layers = nn.Sequential(
            nn.Dropout(p=dropout),
            conv_nd(dims, c, self.out_channels, 1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    
    def _forward(self, x, emb):
        #b, c, *spatial = x.shape
        #x = x.reshape(b, c, -1)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return (self.skip_connection(x) + h)#.reshape(b, c, *spatial)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False, with_xattn=False, xattn_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.with_xattn = with_xattn
        if self.with_xattn:
            if xattn_channels is None:
                xattn_channels = channels
            self.xattn_channels = xattn_channels
            self.qk_x = conv_nd(1, xattn_channels, 2*channels, 1) 
            self.v_x = conv_nd(1, channels, channels, 1)
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x, y=None):
        b, c, *spatial = x.shape
        #assert c==self.channels, f"expected {self.channels} channels, got {c} channels"
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        if y is not None:
            assert self.with_xattn, "y is is only supported as an input for AttentionBlocks with cross attention"
            b, cx, *spatial2 = y.shape
            assert cx==self.xattn_channels, f"expected {self.xattn_channels} channels, got {cx} channels"
            y = y.reshape(b, cx, -1)
            qk = self.qk_x(self.norm(y))
            v = self.v_x(self.norm(h))
            qkv_x = torch.cat([qk,v],dim=-1).reshape(b * self.num_heads, -1, qk.shape[2])
            h = self.attention(qkv_x)+h
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x,y=x_attn)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims=2, mode="nearest"):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.mode = mode
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels_out, channels_out, 3, padding=1)
        if channels_out != channels:
            self.channel_mapper = conv_nd(dims, channels, channels_out, 1)
    def forward(self, x):
        assert x.shape[1] == self.channels

        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims=2):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)
        if channels_out != channels:
            self.channel_mapper = conv_nd(dims, channels, channels_out, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.op(x)
        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        return x

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

if __name__ == "__main__":
    args = OmegaConf.load("./configs/config.yaml")
    unet = create_unet_from_args(args["unet"])
    print(unet)
    im = torch.randn(8, 3, 128, 128)
    mask = torch.randn(8, 1, 128, 128)
    t = torch.zeros([8])
    out = unet(im, mask, t)
    print(out.shape)