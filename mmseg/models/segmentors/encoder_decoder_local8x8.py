import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import build_segmentor
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder_LOCAL8x8(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 fuse_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 global_cfg=None,
                 pretrained=None):
        super(EncoderDecoder_LOCAL8x8, self).__init__()
        self.global_cfg = global_cfg
        self.global_model = build_segmentor(global_cfg.model, train_cfg=global_cfg.train_cfg, test_cfg=global_cfg.test_cfg)

        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_fuse_head(fuse_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        self.global_model.eval()

        for k,v in self.global_model.named_parameters():
            v.requires_grad = False

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_fuse_head(self, fuse_head):
        """Initialize ``fuse_head``"""
        self.fuse_head = builder.build_head(fuse_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder_LOCAL8x8, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

        print('Loading Global Model=======> '+self.global_cfg.global_model_path)
        if not osp.isfile(self.global_cfg.global_model_path):
            raise RuntimeError("========> no checkpoint found at '{}'".format(self.global_cfg.global_model_path))
        global_model_dict = torch.load(self.global_cfg.global_model_path, map_location='cpu')
        self.global_model.load_state_dict(global_model_dict['state_dict'])

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone into a tuple list."""
        x = self.extract_feat(img)
        return x

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in training.
           Generate the LOCAL FEATURE
        """
        losses = dict()
        loss_decode, local_features = self.decode_head.forward_train_with_local_features(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, local_features

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits, local_feature = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits,local_feature

    def _fuse_head_forward_test(self, local_features, global_features):
        """Run forward function and calculate loss for fuse head in
                inference."""
        fuse_logits, _ = self.fuse_head.fuse_forward_test(local_features, global_features)
        return fuse_logits

    def _fuse_features_forward_train(self, local_features, global_features):
        """Run forward function and calculate loss for fuse head in
                inference."""
        _, fuse_features = self.fuse_head.fuse_forward_test(local_features, global_features)
        return fuse_features

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _fuse_head_forward_train(self, local_features, global_features, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_fuse = self.fuse_head.fuse_forward_train(local_features, global_features, gt_semantic_seg)

        losses.update(add_prefix(loss_fuse, 'fuse_edge'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.global_model.eval()
        with torch.no_grad():
            global_features = self.global_model.inference_global_feature(img).detach()
        batch_size, _, h_img, w_img = img.size()
        h_encode = w_encode = int (20 * (h_img/self.backbone.img_size))
        self.h_crop = self.w_crop = self.h_stride = self.w_stride = self.backbone.img_size
        h_grids = max(h_img - self.h_crop + self.h_stride - 1, 0) // self.h_stride + 1
        w_grids = max(w_img - self.w_crop + self.w_stride - 1, 0) // self.w_stride + 1
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * self.h_stride
                x1 = w_idx * self.w_stride
                y2 = min(y1 + self.h_crop, h_img)
                x2 = min(x1 + self.w_crop, w_img)
                y1 = max(y2 - self.h_crop, 0)
                x1 = max(x2 - self.w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.extract_feat(crop_img)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4]
                preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5]
                preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6]
                preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        x = (preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8)
        losses = dict()
        loss_decode, local_features = self._decode_head_forward_train(x, img_metas,gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        fuse_loss = self._fuse_head_forward_train(local_features,global_features, gt_semantic_seg)
        losses.update(fuse_loss)
        return losses

    def inference_global_local_feature(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        global_features = self.global_model.slide_inference_global_features(img, img_meta, rescale)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        x = (preds1, preds2, preds3, preds4)
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)
        fuse_logits = self._fuse_head_forward_test(local_features, global_features)

        return fuse_logits

    def inference_global_local_feature_with_fuse_feature(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        global_features = self.global_model.slide_inference_global_features(img, img_meta, rescale)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0]
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1]
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2]
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3]
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        x = (preds1, preds2, preds3, preds4)
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)
        fuse_features = self._fuse_features_forward_train(local_features, global_features)
        return fuse_features

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        img_crop = img[:, :, 0:h_img - 1, 0:w_img - 1]   # for BSDS500
        #img_crop = img[:, :, 0:h_img - 1, :]   # for NYUD
        batch_size, _, h_crop_img, w_crop_img = img_crop.size()

        global_features_crop = self.global_model.slide_inference_global_features(img_crop, img_meta, rescale).detach()

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        h_grids = max(h_crop_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_crop_img - w_crop + w_stride - 1, 0) // w_stride + 1

        h_encode = int(h_crop_img / 8)
        w_encode = int(w_crop_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_crop_img)
                x2 = min(x1 + w_crop, w_crop_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0].data
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1].data
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2].data
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3].data
                preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4].data
                preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5].data
                preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6].data
                preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7].data
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        x_crop = (preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8)
        local_outs_crop, local_features_crop = self._decode_head_forward_test(x_crop,img_meta)
        fuse_outs_crop = self._fuse_head_forward_test(local_features_crop, global_features_crop)

        fuse_outs = torch.zeros((batch_size,num_classes, h_img, w_img))
        fuse_outs[:, :, 0:h_img - 1, 0:w_img - 1] = fuse_outs_crop    # for BSDS500
        #fuse_outs[:, :, 0:h_img - 1, :] = fuse_outs_crop    # for NYUD
        '''
        if rescale:
            fuse_outs = resize(
                fuse_outs,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        '''
        return fuse_outs

    def slide_inference2(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()

        global_features = self.global_model.slide_inference_global_features(img,img_meta, rescale)

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        h_encode = int(h_img / 8)
        w_encode = int(w_img / 8)
        preds1 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds2 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds3 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds4 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds5 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds6 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds7 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        preds8 = img.new_zeros((batch_size, 256, h_encode, w_encode))
        count_mat = img.new_zeros((batch_size, 1, h_encode, w_encode))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds1[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[0].data
                preds2[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[1].data
                preds3[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[2].data
                preds4[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[3].data
                preds5[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[4].data
                preds6[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[5].data
                preds7[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[6].data
                preds8[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] = crop_seg_logit[7].data
                count_mat[:, :, int(y1 / 8):int(y2 / 8), int(x1 / 8):int(x2 / 8)] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        x = (preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8)
        local_outs, local_features = self._decode_head_forward_test(x,img_meta)
        fuse_outs = self._fuse_head_forward_test(local_features, global_features)
        return fuse_outs

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        '''
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        '''
        return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_pred = self.inference(img, img_meta, rescale)
        #seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        #seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        batch_size, _, h_img, w_img = imgs[0].size()
        seg_logit = torch.zeros([batch_size, 1, h_img, w_img]).cuda()
        img0_crop = imgs[0][:, :, 0:h_img - 1, 0:w_img - 1]
        img0_crop_seg_logit = self.slide_inference2(img0_crop, img_metas[0], rescale)
        seg_logit[:, :, 0:h_img - 1, 0:w_img - 1] = img0_crop_seg_logit
        for i in range(1, len(imgs)):
            #img_cur = imgs[i]
            cur_seg_logit = self.slide_aug_test(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)

        seg_pred = seg_logit.cpu().numpy()
        # unravel batch dim
        #seg_pred = list(seg_pred)
        return seg_pred

    def slide_aug_test(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        batch_size, _, h_img, w_img = img.size()
        if h_img < w_img:
            h_crop = 320
            w_crop =480
            h_stride = 300
            w_stride = 400
        if h_img > w_img:
            h_crop = 480
            w_crop =320
            h_stride = 400
            w_stride = 300
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.slide_inference2(crop_img, img_meta, None)
                preds += F.pad(crop_seg_logit,
                              (int(x1), int(preds.shape[3] - x2), int(y1),
                               int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)

        preds = preds / count_mat
        # '''
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        # '''
        return preds


if __name__ == '__main__':
    model = EncoderDecoder_LOCAL8x8()
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())
