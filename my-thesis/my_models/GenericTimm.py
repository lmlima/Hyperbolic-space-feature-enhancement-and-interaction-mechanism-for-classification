# -*- coding: utf-8 -*-
"""
Autor: Leandro Lima
Email: leandro.m.lima@ufes.br
"""

from torch import nn
from metablock import MetaBlock, MetaBlockSN
from metanet import MetaNet
import torch
import warnings
import timm
from vittimm import MyViTTimm

import torch.nn.functional as F

from densesparse import DenseSparse


class GenericTimm(MyViTTimm):

    def __init__(self, vit, num_class, classifier, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1024, experimental_cfg=None):  # base = 768; huge = 1280

        super(GenericTimm, self).__init__(vit, num_class, neurons_reducer_block, freeze_conv, p_dropout,
                                          comb_method, comb_config, n_feat_conv, experimental_cfg=experimental_cfg)

        freeze = experimental_cfg["late_fusion"]["freeze_backbone"]
        use_softmax = experimental_cfg["late_fusion"]["use_softmax"]

        self.features = vit
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # if classifier is None:
        #     classifier = 'mlp'
        self.classifier_name = classifier

        if n_feat_conv is None:
            crossvit_classes = (timm.models.crossvit.CrossViT)
            if isinstance(self.features, crossvit_classes):
                n_feat_conv = sum(self.features.embed_dim)
            else:
                n_feat_conv = self.features.num_features

        self.use_SNN = experimental_cfg["embedding"]["use_SNN"]
        self.use_DS = experimental_cfg["embedding"]["use_DS"]
        if self.use_DS:
            emb_dim = experimental_cfg["embedding"]["emb_dim"]  # 25
            emb_out = experimental_cfg["embedding"]["emb_out"]  # 15
            cat_max_size = experimental_cfg["embedding"]["categorical_max_size"]  # 15

            col_name = experimental_cfg["embedding"]["col_sparse"]
            col_name_dense = experimental_cfg["embedding"]["col_dense"]

            self.DSTransform = DenseSparse(
                len(col_name_dense), len(col_name),
                emb_dim, emb_out, cat_max_size,
                poincare_dict=experimental_cfg["poincare"],
                use_SNN=self.use_SNN
            )
            if experimental_cfg["embedding"]["use_DDS"]:
                self.DSTransform2 = DenseSparse(
                    len(col_name_dense), len(col_name),
                    emb_dim, emb_out, cat_max_size,
                    poincare_dict=experimental_cfg["poincare"],
                    use_SNN=self.use_SNN
                )
            comb_config = self.DSTransform.output_shape

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metablock' method")
                # comb_div = 32
                comb_div = 1
                while n_feat_conv % (comb_div) != 0:
                    comb_div -= 1

                conv_input_dim = n_feat_conv // comb_div
                if self.use_SNN:
                    self.comb = MetaBlockSN(conv_input_dim, comb_config)  # Normally (40, x)
                else:
                    self.comb = MetaBlock(conv_input_dim, comb_config)  # Normally (40, x)                self.comb_feat_maps = conv_input_dim
                self.comb_div = comb_div
            elif comb_method == 'concat':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'concat' method")
                _n_meta_data = comb_config
                self.comb = 'concat'
            elif comb_method == 'metanet':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metanet' method")
                comb_div = 8
                while n_feat_conv % (comb_div * comb_div) != 0:
                    comb_div -= 1

                conv_input_dim = n_feat_conv // (comb_div * comb_div)
                middle_layer = 64
                self.comb = MetaNet(comb_config, middle_layer, conv_input_dim)  # (n_meta, middle, 20)
                self.comb_feat_maps = conv_input_dim
                self.comb_div = comb_div
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")

            # if self.comb_div is not None:
            #     warnings.warn(F"comb_div = {self.comb_div}")
            # if self.comb_feat_maps is not None:
            #     warnings.warn(F"comb_feat_maps = {self.comb_feat_maps}")
        else:
            self.comb = None

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            if comb_method == 'concat':
                warnings.warn("You're using concat with neurons_reducer_block=0. Make sure you're doing it right!")
            self.reducer_block = None

        if comb_method == 'mat':
            # Projection of meta_data to image features size
            self.data_proj = nn.Linear(_n_meta_data, n_feat_conv)
            # Set _n_meta_data to 0 since MAT merge those extra information in n_feat_conv.
            _n_meta_data = 0

        # Here comes the extra information (if applicable)
        if neurons_reducer_block > 0:
            # self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
            self.classifier = nn.Sequential(
                    nn.Linear(neurons_reducer_block + _n_meta_data, num_class),
                    # nn.Softmax(dim=1)
                )
        else:
            # self.classifier = nn.Linear(n_feat_conv + _n_meta_data, num_class)
            self.classifier = nn.Sequential(
                nn.Linear(n_feat_conv + _n_meta_data, num_class),
                # nn.Softmax(dim=1)
            )

        # freezing the convolution layers
        # features, reducer_block, avg_pooling, classifier, comb
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            if self.reducer_block is not None:
                for param in self.reducer_block.parameters():
                    param.requires_grad = False
            if self.avg_pooling is not None:
                for param in self.avg_pooling.parameters():
                    param.requires_grad = False
            if self.classifier is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = False
            if self.comb is not None:
                for param in self.comb.parameters():
                    param.requires_grad = False

        if use_softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None

        # Late fusion
        self.late_fusion = experimental_cfg["late_fusion"]["late_fusion"]
        self.pre_fusion = experimental_cfg["late_fusion"]["pre_fusion"]
        self.late_residual = experimental_cfg["late_fusion"]["late_residual"]

        if self.late_fusion:
            comb_div = 1
            while n_feat_conv % (comb_div) != 0:
                comb_div -= 1

            conv_input_dim = n_feat_conv // comb_div
            self.late_fusion_fn = MetaBlock(num_class, comb_config)
            if self.pre_fusion:
                self.pre_fusion_fn = MetaBlock(num_class, comb_config)
            self.comb_div = comb_div
            comb_config = 0
        else:
            self.late_fusion_fn = MetaBlock(num_class, comb_config)
            if self.pre_fusion:
                self.pre_fusion_fn = MetaBlock(num_class, comb_config)
            self.comb_div = 1


        # Here comes the extra information (if applicable)
        self.outter_classifier = self.get_classifier(self.classifier_name, comb_config, num_class)


    def get_classifier(self, classifier, size, num_class):
        classifier_size = num_class + size
        if classifier == 'linear':
            return nn.Linear(classifier_size, num_class)
        elif classifier is None:
            return None
        elif classifier == 'mlp':
            hidden_size = 128
            return nn.Sequential(
                nn.Linear(classifier_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_class)
            )
        else:
            raise Exception("There is no classifier called " + classifier + ". Please, check this out.")

    # def forward(self, img, meta_data=None):
    #
    #     # Checking if when passing the metadata, the combination method is set
    #     # if meta_data is not None and self.comb is None:
    #     #     raise Exception("There is no combination method defined but you passed the metadata to the model!")
    #     if meta_data is None and self.comb is not None:
    #         raise Exception("You must pass meta_data since you're using a combination method!")
    #
    #     # x = self.features.forward_features(img)
    #     # x = x if type(x) != tuple else x[0]
    #
    #     if self.outter_classifier is None:
    #         img_class_feat = self.backbone_fusion(img, meta_data=meta_data)
    #         out = self.classifier(img_class_feat)
    #     else:
    #         img_class_feat = self.backbone_fusion(img, meta_data=None)
    #         x = self.classifier(img_class_feat)
    #
    #         if self.softmax:
    #             x = self.softmax(x)
    #
    #         # Concatenate backbone_fusion features with meta_data features
    #         if self.classifier_name == 'mha2':
    #             # x = F.softmax(x, dim=1)
    #             out = self.outter_classifier(x, meta_data)
    #             # out = self.outter_classifier(meta_data, x)
    #
    #         else:
    #             x = torch.cat((x, meta_data), dim=1)
    #
    #             out = self.outter_classifier(x)
    #
    #
    #     # out = self.outter_classifier(x)
    #     # return out if len(out) == 1 else out[0]
    #     return out

    #Forward com late late fusion - experimental
    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        # if meta_data is not None and self.comb is None:
        #     raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # x = self.features.forward_features(img)
        # x = x if type(x) != tuple else x[0]

        if self.outter_classifier is None:
            img_class_feat = self.backbone_fusion(img, meta_data=meta_data)
            out = self.classifier(img_class_feat)
        else:
            img_class_feat = self.backbone_fusion(img, meta_data=None)
            x = self.classifier(img_class_feat)

            if self.softmax:
                x = self.softmax(x)

            if self.use_DS:
                meta_data = self.DSTransform(meta_data)

            # Concatenate backbone_fusion features with meta_data features
            if self.classifier_name == 'mha2':
                # x = F.softmax(x, dim=1)
                out = self.outter_classifier(x, meta_data)
                # out = self.outter_classifier(meta_data, x)

            else:
                if not self.pre_fusion:
                    x_orig = x
                    x = torch.cat((x, meta_data), dim=1)
                else:
                    x_orig = x

                    x = x.view(x.size(0), -1, self.comb_div).squeeze(-1)
                    # x = x.view(x.size(0), -1, 32).squeeze(-1) # getting the feature maps

                    # Make sure there is at least 3 dimensions, only MetaBlock
                    # if len(x.shape) < 3:
                    #     x = x.unsqueeze(2)
                    x = self.pre_fusion_fn(x, meta_data).squeeze(-1)
                    x = torch.cat((x, meta_data), dim=1)

                out = self.outter_classifier(x)

                if self.late_residual:
                    x_lf = x_orig.view(x_orig.size(0), -1, self.comb_div).squeeze(-1)
                    # x = x.view(x.size(0), -1, 32).squeeze(-1) # getting the feature maps

                    # Make sure there is at least 3 dimensions
                    if len(x_lf.shape) < 3:
                        x_lf = x_lf.unsqueeze(2)
                    x_lf = self.late_fusion_fn(x_lf, meta_data).squeeze(-1)

                    out = out + x_lf

        # out = self.outter_classifier(x)
        # return out if len(out) == 1 else out[0]
        return out

