from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import models


DEVICE = torch.device("cuda")


def softmax_aggregate(predicted_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx + 1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in
                       enumerate(object_ids)}


def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad + 1) // 2, width_pad // 2, (height_pad + 1) // 2, height_pad // 2]
    return padding


def apply_padding(x, y, padding):
    B, L, C, H, W = x.size()
    x = x.view(B * L, C, H, W)
    x = F.pad(x, padding, mode='reflect')
    _, _, height, width = x.size()
    x = x.view(B, L, C, height, width)
    y = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in y]
    return x, y


def unpad(tensor, padding):
    if isinstance(tensor, (dict, OrderedDict)):
        return {key: unpad(val, padding) for key, val in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [unpad(elem, padding) for elem in tensor]
    else:
        _, _, _, height, width = tensor.size()
        tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
        return tensor

class LinearRelu(nn.Sequential):
    def __init__(self, *linear_args):
        super().__init__()
        self.add_module('linear', nn.Linear(*linear_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size_tensor = input_tensor.size(0)
        x_dim = input_tensor.size(3)
        y_dim = input_tensor.size(2)

        xx_ones = torch.ones([1, x_dim], dtype=torch.int32)
        xx_range = torch.arange(y_dim, dtype=torch.int32).unsqueeze(1)
        xx_channel = torch.matmul(xx_range, xx_ones).unsqueeze(0)

        yy_ones = torch.ones([y_dim, 1], dtype=torch.int32)
        yy_range = torch.arange(x_dim, dtype=torch.int32).unsqueeze(0)
        yy_channel = torch.matmul(yy_ones, yy_range).unsqueeze(0)

        xx_channel = xx_channel.float() / (y_dim - 1)
        yy_channel = yy_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class REFINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s16 = ConvRelu(2048, 256, 1, 1, 0)
        self.blend_s16 = ConvRelu(261, 128, 3, 1, 1)
        self.conv_s8 = ConvRelu(512, 128, 1, 1, 0)
        self.blend_s8 = ConvRelu(130, 128, 3, 1, 1)
        self.conv_s4 = ConvRelu(256, 128, 1, 1, 0)
        self.blend_s4 = ConvRelu(130, 128, 3, 1, 1)
        self.deconv1_1 = nn.ConvTranspose2d(128, 2, 4, 2, 1, bias=True)
        self.deconv1_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(128, 2, 4, 2, 1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(128, 2, 3, 1, 1, bias=True)
        self.predictor = nn.ConvTranspose2d(6, 2, 6, 4, 1, bias=True)
        self.coord = AddCoords(with_r=True)

    def forward(self, feats, state):
        prev_seg = state['prev_seg']
        clue = self.coord(prev_seg)
        u = torch.cat([self.conv_s16(feats['s16']), clue], dim=-3)
        u = self.blend_s16(u)
        out_16 = self.deconv1_1(u)

        u = torch.cat([self.conv_s8(feats['s8']), out_16], dim=-3)
        u = self.blend_s8(u)
        out_8 = self.deconv2(u)

        u = torch.cat([self.conv_s4(feats['s4']), out_8], dim=-3)
        u = self.blend_s4(u)
        out_4 = self.deconv3(u)

        segscore = self.predictor(torch.cat([self.deconv1_2(out_16), out_8, out_4], dim=1))

        return segscore


class VOS(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()
        self.backbone = getattr(models.backbones, backbone_cfg[0])(*backbone_cfg[1])
        self.refine = REFINE()

    def get_init_state(self, img, given_seg):
        state = {}
        state['prev_seg'] = given_seg
        return state

    def update(self, feats, pred_seg, state):
        state['prev_seg'] = pred_seg
        return state

    def extract_feats(self, img):
        feats = self.backbone.get_features(img)
        return feats

    def forward(self, feats, state):
        segscore = self.refine(feats, state)
        return state, segscore


