import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from functools import partial
import torch.utils.checkpoint as checkpoint
try:
    from huggingface_hub import hf_hub_download
except:
    hf_hub_download = None      # install huggingface_hub if you would like to download models conveniently from huggingface



class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))





default_UniRepLKNet_A_F_P_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_N_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_T_kernel_sizes = ((3, 3, 3),
                                      (13, 13, 13),
                                      (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                      (13, 13, 13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = ((3, 3, 3),
                                             (13, 13, 13),
                                             (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                             (13, 13, 13))
default_CogRepLKNet_kernel_sizes = ((3, 3),
                                    (13, 13),
                                    (13, 13, 13),
                                    (13, 13))
v2_CogRepLKNet_kernel_sizes = ((5, 5, 5),
                                 (15, 15, 15),
                                 (15, 5, 5, 15, 5, 5),
                                 (15, 15))

UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)
CogRepLKNet_depths = (2, 2, 3, 2)
CogRepLKNet_v2_depths = (3, 3, 6, 2)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes,
    CogRepLKNet_depths: default_CogRepLKNet_kernel_sizes,
    CogRepLKNet_v2_depths: v2_CogRepLKNet_kernel_sizes
}



class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#   For easy use as backbone in MMDetection framework. Ignore these lines if you do not use MMDetection
if has_mmdet:
    @det_BACKBONES.register_module()
    class UniRepLKNetBackbone(UniRepLKNet):
        def __init__(self,
                     depths=(3, 3, 27, 3),
                     dims=(96, 192, 384, 768),
                     drop_path_rate=0.,
                     layer_scale_init_value=1e-6,
                     kernel_sizes=None,
                     deploy=False,
                     with_cp=False,
                     init_cfg=None,
                     attempt_use_lk_impl=False):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, deploy=deploy, with_cp=with_cp,
                             init_cfg=init_cfg, attempt_use_lk_impl=attempt_use_lk_impl, use_sync_bn=True)

#   For easy use as backbone in MMSegmentation framework. Ignore these lines if you do not use MMSegmentation
if has_mmseg:
    @seg_BACKBONES.register_module()
    class UniRepLKNetBackbone(UniRepLKNet):
        def __init__(self,
                     depths=(3, 3, 27, 3),
                     dims=(96, 192, 384, 768),
                     drop_path_rate=0.,
                     layer_scale_init_value=1e-6,
                     kernel_sizes=None,
                     deploy=False,
                     with_cp=False,
                     init_cfg=None,
                     attempt_use_lk_impl=False):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, deploy=deploy, with_cp=with_cp,
                             init_cfg=init_cfg, attempt_use_lk_impl=attempt_use_lk_impl, use_sync_bn=True)


model_urls = {
    #TODO: it seems that google drive does not support direct downloading with url? so where to upload the checkpoints other than huggingface? any suggestions?
}

huggingface_file_names = {
    "unireplknet_a_1k": "unireplknet_a_in1k_224_acc77.03.pth",
    "unireplknet_f_1k": "unireplknet_f_in1k_224_acc78.58.pth",
    "unireplknet_p_1k": "unireplknet_p_in1k_224_acc80.23.pth",
    "unireplknet_n_1k": "unireplknet_n_in1k_224_acc81.64.pth",
    "unireplknet_t_1k": "unireplknet_t_in1k_224_acc83.21.pth",
    "unireplknet_s_1k": "unireplknet_s_in1k_224_acc83.91.pth",
    "unireplknet_s_22k": "unireplknet_s_in22k_pretrain.pth",
    "unireplknet_s_22k_to_1k": "unireplknet_s_in22k_to_in1k_384_acc86.44.pth",
    "unireplknet_b_22k": "unireplknet_b_in22k_pretrain.pth",
    "unireplknet_b_22k_to_1k": "unireplknet_b_in22k_to_in1k_384_acc87.40.pth",
    "unireplknet_l_22k": "unireplknet_l_in22k_pretrain.pth",
    "unireplknet_l_22k_to_1k": "unireplknet_l_in22k_to_in1k_384_acc87.88.pth",
    "unireplknet_xl_22k": "unireplknet_xl_in22k_pretrain.pth",
    "unireplknet_xl_22k_to_1k": "unireplknet_xl_in22k_to_in1k_384_acc87.96.pth"
}

def load_with_key(model, key):
    # if huggingface hub is found, download from our huggingface repo
    if hf_hub_download is not None:
        repo_id = 'DingXiaoH/UniRepLKNet'
        cache_file = hf_hub_download(repo_id=repo_id, filename=huggingface_file_names[key])
        checkpoint = torch.load(cache_file, map_location='cpu')
    else:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls[key], map_location="cpu", check_hash=True)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)

def initialize_with_pretrained(model, model_name, in_1k_pretrained, in_22k_pretrained, in_22k_to_1k):
    if in_1k_pretrained:
        key = model_name + '_1k'
    elif in_22k_pretrained:
        key = model_name + '_22k'
    elif in_22k_to_1k:
        key = model_name + '_22k_to_1k'
    else:
        key = None
    if key:
        load_with_key(model, key)

@register_model
def unireplknet_a(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(40, 80, 160, 320), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_a', in_1k_pretrained, False, False)
    return model


@register_model
def CogRepLKNet(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=CogRepLKNet_depths, dims=(48, 96, 192, 384), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_xl', False, in_1k_pretrained, False)
    return model

@register_model
def CogRepLKNet_v2(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=CogRepLKNet_v2_depths, dims=(80, 160, 320, 640), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_xl', False, in_1k_pretrained, False)
    return model



if __name__ == '__main__':
    #   Test case showing the equivalency of Structural Re-parameterization
    x = torch.randn(2, 4, 19, 19)
    layer = UniRepLKNetBlock(4, kernel_size=13, attempt_use_lk_impl=False)
    for n, p in layer.named_parameters():
        if 'beta' in n:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.normal_(p)
    for n, p in layer.named_buffers():
        if 'running_var' in n:
            print('random init var')
            torch.nn.init.uniform_(p)
            p.data += 2
        elif 'running_mean' in n:
            print('random init mean')
            torch.nn.init.uniform_(p)
    layer.gamma.data += 0.5
    layer.eval()
    origin_y = layer(x)
    layer.reparameterize()
    eq_y = layer(x)
    print(layer)
    print(eq_y - origin_y)
    print((eq_y - origin_y).abs().sum() / origin_y.abs().sum())
