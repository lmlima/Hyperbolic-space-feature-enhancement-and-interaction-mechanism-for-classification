# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Function to load the CNN models
"""

from effnet import MyEffnet
from densenet import MyDensenet
from mobilenet import MyMobilenet
from resnet import MyResnet
from vggnet import MyVGGNet
from inceptionv4 import MyInceptionV4
from senet import MySenet
from vit import MyViT
from vittimm import MyViTTimm
from torchvision import models
import pretrainedmodels as ptm
from efficientnet_pytorch import EfficientNet
from transformers import ViTModel, BeitModel, DeiTModel
import timm
from GenericTimm import GenericTimm
from wrn_model import WideResNet

_MODELS = ['resnet-50', 'resnet-101', 'densenet-121', 'inceptionv4', 'googlenet', 'vgg-13', 'vgg-16', 'vgg-19',
           'mobilenet', 'efficientnet-b4', 'senet', 'vit', 'deit', 'beit', "pit"]

import torch.nn as nn


def get_norm_and_size(model_name):
    if model_name.split("_")[0] == 'GenericTimm':
        model_name = "_".join(model_name.split("_")[1:])    # Reconstruct the model name

    if model_name == "inceptionv4":
        return [229, 229], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif model_name in ["beit", "vit"]:
        return [224, 224], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif model_name in ['resnet-50', 'resnet-101', 'densenet-121', 'googlenet', 'vgg-13', 'vgg-16', 'vgg-19',
                        'mobilenet', 'efficientnet-b4', 'senet', 'vit', 'deit']:
        return [224, 224], ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        # Load pre defined parameters for timm models
        pre_load_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        size = list(pre_load_model.default_cfg["input_size"][1:])
        mean_std = (list(pre_load_model.default_cfg["mean"]), list(pre_load_model.default_cfg["std"]))
        del pre_load_model
        return size, mean_std
        # return [384, 384], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def set_model(model_name, num_class, neurons_reducer_block=0, comb_method=None, comb_config=None, pretrained=True,
              freeze_conv=False, p_dropout=0.5, model_backbone=None, classifier=None, experimental_cfg=None):
    """
    Function to load the all models
    :param model_name:
    :param num_class:
    :param neurons_reducer_block:
    :param comb_method:
    :param comb_config:
    :param pretrained:
    :param freeze_conv:
    :param p_dropout:
    :param model_backbone: Optional. A preloaded TIMM model backbone. If None, a 'model_name' backbone will be loaded
    using TIMM libary.
    :return: Model
    """
    if pretrained:
        pre_ptm = 'imagenet'
        pre_torch = True
    else:
        pre_torch = False
        pre_ptm = None

    # if model_name not in _MODELS:
    #     raise Exception("The model {} is not available!".format(model_name))

    model = None
    if model_name == 'resnet-50':
        model = MyResnet(models.resnet50(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'resnet-101':
        model = MyResnet(models.resnet101(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'densenet-121':
        model = MyDensenet(models.densenet121(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                           comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-13':
        model = MyVGGNet(models.vgg13_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-16':
        model = MyVGGNet(models.vgg16_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-19':
        model = MyVGGNet(models.vgg19_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'mobilenet':
        model = MyMobilenet(models.mobilenet_v2(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                            comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'efficientnet-b4':
        if pretrained:
            model = MyEffnet(EfficientNet.from_pretrained(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyEffnet(EfficientNet.from_name(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'inceptionv4':
        model = MyInceptionV4(ptm.inceptionv4(num_classes=1000, pretrained=pre_ptm), num_class, neurons_reducer_block,
                              freeze_conv, comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'senet':
        model = MySenet(ptm.senet154(num_classes=1000, pretrained=pre_ptm), num_class, neurons_reducer_block,
                        freeze_conv, comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vit':
        pretrained_vit = 'google/vit-base-patch16-224-in21k'  # Default
        # pretrained_vit = "facebook/dino-vitb16" # Interpretability
        # pretrained_vit = 'vit_large_patch16_384'  # Performance
        # pretrained_vit = "google/vit-huge-patch14-224-in21k" # Too large, memory issues
        model = MyViT(ViTModel.from_pretrained(pretrained_vit), num_class, neurons_reducer_block, freeze_conv,
                      comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'deit':
        pretrained_vit = 'facebook/deit-base-distilled-patch16-224'  # Default
        model = MyViT(DeiTModel.from_pretrained(pretrained_vit), num_class, neurons_reducer_block, freeze_conv,
                      comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'beit':
        # pretrained_vit = 'microsoft/beit-base-patch16-224' # Default
        pretrained_vit = "microsoft/beit-base-patch16-224-pt22k-ft22k"  # Recommended
        model = MyViT(BeitModel.from_pretrained(pretrained_vit), num_class, neurons_reducer_block, freeze_conv,
                      comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'pit':
        pretrained_vit = "pit_b_distilled_224"  # Recommended
        # num_classes=0, global_pool='' to create with no classifier and pooling
        model = MyViTTimm(timm.create_model(pretrained_vit, pretrained=True, num_classes=0), num_class,
                          neurons_reducer_block, freeze_conv,
                          comb_method=comb_method, comb_config=comb_config)
    elif model_name.split("_")[0] == 'GenericTimm':  # Ex. "GenericTimm_mobilenetv2_120d"
        pt = "_".join(model_name.split("_")[1:])    # Reconstruct the model name
        if model_backbone is None:
            model_pt = timm.create_model(pt, pretrained=True, num_classes=0)
        else:
            model_pt = model_backbone
        model = GenericTimm(model_pt, num_class, classifier=classifier, neurons_reducer_block=neurons_reducer_block, n_feat_conv=None,
                          comb_method=comb_method, comb_config=comb_config, experimental_cfg=experimental_cfg)
    else:
        pt = model_name
        if model_backbone is None:
            model_pt = timm.create_model(pt, pretrained=True, num_classes=0)
        else:
            model_pt = model_backbone
        model = MyViTTimm(model_pt, num_class, neurons_reducer_block, n_feat_conv=None,
                          comb_method=comb_method, comb_config=comb_config, experimental_cfg=experimental_cfg)
    return model


class FSLWrapper(nn.Module):

    def __init__(
            self,
            model,
            use_fc: bool = False
    ):
        super(FSLWrapper, self).__init__()
        self.backbone = model
        self.use_fc = use_fc

    def forward(self, x, meta=None):
        if self.use_fc:
            if isinstance(self.backbone, WideResNet):
                out = self.backbone(x, meta)[1]
            else:
                out = self.backbone(x, meta)

        else:
            if isinstance(self.backbone, WideResNet):
                out = self.backbone(x, meta)[0]
            else:
                out = self.backbone.backbone_fusion(x, meta)

        return out

    def set_use_fc(self, use_fc: bool):
        """
        Change the use_fc property. Allow to decide when and where the model should use its last
        fully connected layer.
        Args:
            use_fc: whether to set self.use_fc to True or False
        """
        self.use_fc = use_fc
