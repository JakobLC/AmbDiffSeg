import math
import torch
import torch.nn as nn

def convert_module_to_f16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, 
                 emb_dim=0,
                 pool=True,norm=False, mc_dropout=False, dropout_rate=0.0):
        super(DownConvBlock, self).__init__()
        layers1 = []
        layers2 = []
        self.mc_dropout = mc_dropout

        if pool:
            layers1.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers1.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers1.append(nn.ReLU(inplace=False))
        layers1.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

        if emb_dim > 0:
            self.use_emb = True
            self.emb = nn.Sequential(nn.ReLU(inplace=False),
                                     nn.Linear(emb_dim, output_dim))
        else:
            self.use_emb = False

        layers2.append(nn.ReLU(inplace=False))
        layers2.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers2.append(nn.ReLU(inplace=False))

        if norm:
            layers2.append(nn.BatchNorm2d(output_dim))

        if self.mc_dropout is True:
            self.dropout_op = nn.Dropout(p=dropout_rate)

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)

        self.layers1.apply(init_weights)
        self.layers2.apply(init_weights)

    def forward(self, patch, emb=None):
        out = self.layers1(patch)
        if emb is not None:
            assert self.use_emb
            out = out + self.emb(emb)[:,:,None,None]
        out = self.layers2(out)
        if self.mc_dropout is True:
            out = self.dropout_op(out)
        return out


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding,
                 emb_dim=0,
                 bilinear=True,norm=False, mc_dropout=False, dropout_rate=0.0):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear
        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)
#        pdb.set_trace()
        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, 
                                        emb_dim=emb_dim,
                                        pool=False,norm=norm, mc_dropout=mc_dropout, dropout_rate=dropout_rate)

    def forward(self, x, bridge, emb=None):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out, emb)

        return out


def timestep_embedding(timesteps, dim, max_period=10):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GenProbUNet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels=4, 
                 num_classes=1, 
                 num_filters=[32, 64, 128, 192, 320], 
                 initializers={'w':'he_normal', 'b':'normal'}, 
                 apply_last_layer=True, 
                 padding=True,
                 norm=True, 
                 mc_dropout=False, 
                 dropout_rate=0.0,
                 use_embedding=True,
                 final_act="tanh"):
        super(GenProbUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()
        self.use_emb = use_embedding

        self.fp16_attrs = ['contracting_path', 'upsampling_path']

        if self.use_emb:
            self.model_channels = self.num_filters[0]
            self.time_embed = nn.Sequential(
                nn.Linear(self.model_channels, self.model_channels*4),
                nn.ReLU(),
                nn.Linear(self.model_channels*4, self.model_channels*4),
            )
            emb_dim = self.model_channels*4
            self.fp16_attrs.append('time_embed')
        else:
            emb_dim = 0
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input,
                                                       output,
                                                       initializers,
                                                       padding,
                                                       emb_dim=emb_dim,
                                                       pool=pool,
                                                       norm=norm,
                                                       mc_dropout=mc_dropout,
                                                       dropout_rate=dropout_rate))


        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2

        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]

            if i == 0:
                norm = False
            else:
                norm = norm

            self.upsampling_path.append(UpConvBlock(input,
                                                    output,
                                                    initializers,
                                                    padding,
                                                    emb_dim=emb_dim,
                                                    norm=norm,
                                                    mc_dropout=mc_dropout,
                                                    dropout_rate=dropout_rate))
        assert final_act in ["tanh", "none"], "final_act must be either 'tanh' or 'none'"
        if self.apply_last_layer:
            self.last_layer = nn.Sequential(nn.Conv2d(output, num_classes, kernel_size=1),
                                            nn.Tanh() if final_act == "tanh" else nn.Identity())
            self.fp16_attrs.append('last_layer')

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.contracting_path.parameters()).dtype

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

    def get_pred(self, patch, sigmoid=True):
        s = torch.sigmoid if sigmoid else lambda x: x
        return s(self.forward(patch))

    def forward(self, image, sample, timesteps=None):
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(len(sample))
        x = torch.cat([sample,image],1).type(self.inner_dtype)
        if timesteps is not None:
            assert self.use_emb, "Time embedding is not enabled, but timesteps are provided"
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).type(self.inner_dtype))
        else:
            assert not self.use_emb, "Need to provide timesteps for time embedding"
            emb = None
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x,emb)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1],emb)

        del blocks

        if self.apply_last_layer:
            x =  self.last_layer(x).type(sample.dtype)

        return x


def main():
    model = GenProbUNet()
    t = torch.rand(3)
    x = torch.randn(3, 2, 128, 128)
    print(model(x,t).shape)
    import jlc
    jlc.num_of_params(model)


if __name__=="__main__":
    main()