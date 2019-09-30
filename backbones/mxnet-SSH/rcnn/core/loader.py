from __future__ import print_function
import sys
import mxnet as mx
import numpy as np
import cv2
from mxnet.executor_manager import _split_input_slice

from rcnn.config import config
from rcnn.io.image import tensor_vstack
from rcnn.io.rpn import get_rpn_testbatch, get_rpn_batch, assign_anchor, assign_anchor_fpn, get_crop_batch
#from rcnn.io.rcnn import get_rcnn_testbatch, get_rcnn_batch
from rcnn.io.rcnn import get_rcnn_testbatch, get_fpn_rcnn_testbatch, get_fpn_maskrcnn_batch


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=1, shuffle=False,
                 has_rpn=False, withlabel=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn
        self.withlabel = withlabel

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info'] #TODO
            #self.data_name = ['data']
        else:
            self.data_name = ['data', 'rois']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, \
                   mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            if not self.withlabel:
              data, label = get_rpn_testbatch(roidb)
            else:
              data, label = get_rpn_batch(roidb)
        else:
            data, label = get_rcnn_testbatch(roidb)
        #print('in_loader', label['gt_boxes'].shape, file=sys.stderr)
        self.im_info = data['im_info']
        self.data = [mx.nd.array(data[name]) for name in self.data_name]


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, ctx=None, work_load_list=None, aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                # Avoid putting different aspect ratio image into the same bucket,
                # which may cause bucketing warning.
                pad_horz = self.batch_size - len(horz_inds) % self.batch_size
                pad_vert = self.batch_size - len(vert_inds) % self.batch_size
                horz_inds = np.hstack([horz_inds, horz_inds[:pad_horz]])
                vert_inds = np.hstack([vert_inds, vert_inds[:pad_vert]])
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                inds = np.reshape(inds[:], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get each device
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rcnn_batch(iroidb)
            data_list.append(data)
            label_list.append(label)

        all_data = dict()
        for key in data_list[0].keys():
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_list[0].keys():
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]


class AnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_batch(iroidb)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]

        new_label_list = []
        for data, label in zip(data_list, label_list):
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

            # assign anchor for label
            label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'],
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border)
            new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

class AnchorLoaderFPN(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(AnchorLoaderFPN, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        #self.feat_stride = feat_stride
        #self.anchor_scales = anchor_scales
        #self.anchor_ratios = anchor_ratios
        #self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.feat_stride = config.RPN_FEAT_STRIDE

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        #self.data_name = ['data']
        #self.label_name = []
        #self.label_name.append('label')
        #self.label_name.append('bbox_target')
        #self.label_name.append('bbox_weight')

        self.data_name = ['data']
        if config.TRAIN.RPN_ENABLE_OHEM<2:
          self.label_name = ['label', 'bbox_target', 'bbox_weight']
        else:
          #self.label_name = ['label', 'bbox_target', 'bbox_weight']
          self.label_name = []
          names = ['label', 'bbox_target', 'bbox_weight']
          #names = ['label', 'bbox_weight']
          for stride in self.feat_stride:
            for n in names:
              k = "%s_stride%d"%(n,stride)
              self.label_name.append(k)
        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.visid = -1

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()
        #self.visid = 0

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                # Avoid putting different aspect ratio image into the same bucket,
                # which may cause bucketing warning.
                pad_horz = self.batch_size - len(horz_inds) % self.batch_size
                pad_vert = self.batch_size - len(vert_inds) % self.batch_size
                horz_inds = np.hstack([horz_inds, horz_inds[:pad_horz]])
                vert_inds = np.hstack([vert_inds, vert_inds[:pad_vert]])
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))

                inds = np.reshape(inds[:], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        dummy_boxes = np.zeros((0, 5))
        dummy_info = [ [max_shapes['data'][2], max_shapes['data'][3], 1.0] ]

        # infer shape
        feat_shape_list = []
        for i in range(len(self.feat_stride)):
            _, feat_shape, _ = self.feat_sym[i].infer_shape(**max_shapes)
            feat_shape = [int(i) for i in feat_shape[0]]
            feat_shape_list.append(feat_shape)

        label_dict = assign_anchor_fpn(feat_shape_list, dummy_boxes, dummy_info)
        label_list = []
        for k in self.label_name:
          label_list.append(label_dict[k])
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label_list)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_batch(iroidb)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for i_card in range(len(data_list)):
            data_list[i_card]['data'] = data_tensor[
                                        i_card * config.TRAIN.BATCH_IMAGES:(1 + i_card) * config.TRAIN.BATCH_IMAGES]


        for data, label in zip(data_list, label_list):
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            feat_shape_list = []
            for s in range(len(self.feat_stride)):
                _, feat_shape, _ = self.feat_sym[s].infer_shape(**data_shape)
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shape_list.append(feat_shape)
            for k in self.label_name:
              label[k] = [0 for i in range(config.TRAIN.BATCH_IMAGES)]

            for im_i in range(config.TRAIN.BATCH_IMAGES):
                #im_info = data['im_info'][im_i]
                #gt_boxes = label['gt_boxes'][im_i][0]
                im_info = data['im_info']
                gt_boxes = label['gt_boxes']
                #print('im_info', im_info.shape)
                #print(gt_boxes.shape)
                vis=True if self.visid>=0 else False
                label_dict = assign_anchor_fpn(feat_shape_list, gt_boxes, im_info, vis)
                #do visualize for debug here
                if self.visid>=0 and self.visid<10:
                  for _roi in roidb:
                    print('image', _roi['image'], file=sys.stderr)
                  self.visid+=1
                  anchors = label_dict['anchors'].copy()
                  _im = data['data'].copy()
                  #print(label_dict['olabel'].shape)
                  _label = label_dict['olabel'].copy()
                  _gt_boxes = gt_boxes.copy().astype(np.int)
                  filename = './vis/A%d.png'%self.visid
                  _im = _im[0].transpose( (1,2,0) )
                  _im = _im[...,::-1] #bgr
                  for c in range(3):
                    _im[:,:,c] += config.PIXEL_MEANS[c]
                  _im = _im.astype(np.uint8).copy()
                  fg_inds = np.where(_label == 1)[0]
                  print(_im.shape, _label.shape, anchors.shape, len(fg_inds), _gt_boxes.shape, file=sys.stderr)
                  #draw FG anchors
                  _bc = 0
                  for a in range(anchors.shape[0]):
                    anchor = anchors[a].astype(np.int)
                    l = _label[a]
                    if l!=1:
                      continue
                    #print('drawing', _im.shape, anchor)
                    cv2.rectangle(_im, (anchor[0], anchor[1]), (anchor[2], anchor[3]), (255, 0, 0), 1)
                    _bc+=1
                  for a in range(_gt_boxes.shape[0]):
                    _box = _gt_boxes[a]
                    cv2.rectangle(_im, (_box[0], _box[1]), (_box[2], _box[3]), (0, 0, 255), 1)
                  print('draw to', filename, _bc, file=sys.stderr)
                  cv2.imwrite(filename, _im)


                for k in self.label_name:
                  #print('0in_loader', k, label_dict[k].shape, file=sys.stderr)
                  label[k][im_i] = label_dict[k]
            for k in self.label_name:
              label[k] = np.vstack(label[k])
              #print('in_loader', k, label[k].shape, file=sys.stderr)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = 0 if key.startswith('bbox_') else -1
            #print('label vstack', key, pad, len(label_list), file=sys.stderr)
            all_label[key] = tensor_vstack([batch[key] for batch in label_list], pad=pad)
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]


class CropLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(CropLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        #self.feat_stride = feat_stride
        #self.anchor_scales = anchor_scales
        #self.anchor_ratios = anchor_ratios
        #self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.feat_stride = config.RPN_FEAT_STRIDE

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        #self.data_name = ['data']
        #self.label_name = []
        #self.label_name.append('label')
        #self.label_name.append('bbox_target')
        #self.label_name.append('bbox_weight')

        self.data_name = ['data']
        if config.TRAIN.RPN_ENABLE_OHEM<0:
          self.label_name = ['label', 'bbox_target', 'bbox_weight']
        else:
          #self.label_name = ['label', 'bbox_target', 'bbox_weight']
          self.label_name = []
          names = ['label', 'bbox_target', 'bbox_weight']
          #names = ['label', 'bbox_weight']
          for stride in self.feat_stride:
            for n in names:
              k = "%s_stride%d"%(n,stride)
              self.label_name.append(k)
        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
          np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        dummy_boxes = np.zeros((0, 5))
        dummy_info = [ [max_shapes['data'][2], max_shapes['data'][3], 1.0] ]

        # infer shape
        feat_shape_list = []
        for i in range(len(self.feat_stride)):
            _, feat_shape, _ = self.feat_sym[i].infer_shape(**max_shapes)
            feat_shape = [int(i) for i in feat_shape[0]]
            feat_shape_list.append(feat_shape)

        label_dict = assign_anchor_fpn(feat_shape_list, dummy_boxes, dummy_info)
        label_list = []
        for k in self.label_name:
          label_list.append(label_dict[k])
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label_list)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        assert cur_to==cur_from+self.batch_size
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_crop_batch(iroidb)
            data_list += data
            label_list += label
            #data_list.append(data)
            #label_list.append(label)

        # pad data first and then assign anchor (read label)
        #data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        #for i_card in range(len(data_list)):
        #    data_list[i_card]['data'] = data_tensor[
        #                                i_card * config.TRAIN.BATCH_IMAGES:(1 + i_card) * config.TRAIN.BATCH_IMAGES]

        for data, label in zip(data_list, label_list):
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            feat_shape_list = []
            for s in range(len(self.feat_stride)):
                _, feat_shape, _ = self.feat_sym[s].infer_shape(**data_shape)
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shape_list.append(feat_shape)
            #for k in self.label_name:
            #  label[k] = [0 for i in range(config.TRAIN.BATCH_IMAGES)]
            im_info = data['im_info']
            gt_boxes = label['gt_boxes']
            #print('im_info', im_info.shape)
            #print(gt_boxes.shape)
            label_dict = assign_anchor_fpn(feat_shape_list, gt_boxes, im_info)
            for k in self.label_name:
              label[k] = label_dict[k]

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = 0 if key.startswith('bbox_') else -1
            #print('label vstack', key, pad, len(label_list), file=sys.stderr)
            all_label[key] = tensor_vstack([batch[key] for batch in label_list], pad=pad)
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]


class MaskROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, ctx=None, work_load_list=None, aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: ROIIter
        """
        super(MaskROIIter, self).__init__()
        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data']
        self.label_name = []
        for s in config.RCNN_FEAT_STRIDE:
            self.data_name.append('rois_stride%s' % s)
            self.label_name.append('label_stride%s' % s)
            self.label_name.append('bbox_target_stride%s' % s)
            self.label_name.append('bbox_weight_stride%s' % s)
        for s in config.RCNN_FEAT_STRIDE:
            self.label_name.append('mask_target_stride%s' % s)
            self.label_name.append('mask_weight_stride%s' % s)

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                # Avoid putting different aspect ratio image into the same bucket,
                # which may cause bucketing warning.
                pad_horz = self.batch_size - len(horz_inds) % self.batch_size
                pad_vert = self.batch_size - len(vert_inds) % self.batch_size
                horz_inds = np.hstack([horz_inds, horz_inds[:pad_horz]])
                vert_inds = np.hstack([vert_inds, vert_inds[:pad_vert]])
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))

                inds = np.reshape(inds[:], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        im_array_list = []
        levels_data_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            im_array, levels_data = get_fpn_maskrcnn_batch(iroidb)
            im_array_list.append(im_array)
            levels_data_list.append(levels_data)

        all_data, all_label = self._make_data_and_labels(im_array_list, levels_data_list)
        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]

    def _make_data_and_labels(self, im_array_list, levels_data_list):
        data_list = []
        label_list = []

        rois_num_on_levels = {'stride%s' % s: 0 for s in config.RCNN_FEAT_STRIDE}

        for s in config.RCNN_FEAT_STRIDE:
            max_rois_num = 0
            for levels_data in levels_data_list:
                for im_i in levels_data:
                    rois_num = levels_data[im_i]['rois_on_levels']['stride%s' % s].shape[0]
                    max_rois_num = max(rois_num, max_rois_num)
            rois_num_on_levels['stride%s' % s] = max_rois_num

        # align to num_imgs
        num_imgs = len(levels_data_list[0])
        for s in config.RCNN_FEAT_STRIDE:
            if rois_num_on_levels['stride%s' % s] == 0:
                rois_num_on_levels['stride%s' % s] = num_imgs
                continue
            if rois_num_on_levels['stride%s' % s] % num_imgs != 0:
                ex = num_imgs - rois_num_on_levels['stride%s' % s] % num_imgs
                rois_num_on_levels['stride%s' % s] += ex

        for im_array, data_on_imgs in zip(im_array_list, levels_data_list):
            num_imgs = len(data_on_imgs)
            for s in config.RCNN_FEAT_STRIDE:
                bucket_size = rois_num_on_levels['stride%s' % s]
                for im_i in range(num_imgs):
                    _rois = data_on_imgs['img_%s' % im_i]['rois_on_levels']['stride%s' % s]
                    _labels = data_on_imgs['img_%s' % im_i]['labels_on_levels']['stride%s' % s]
                    _bbox_targets = data_on_imgs['img_%s' % im_i]['bbox_targets_on_levels']['stride%s' % s]
                    _bbox_weights = data_on_imgs['img_%s' % im_i]['bbox_weights_on_levels']['stride%s' % s]
                    _mask_targets = data_on_imgs['img_%s' % im_i]['mask_targets_on_levels']['stride%s' % s]
                    _mask_weights = data_on_imgs['img_%s' % im_i]['mask_weights_on_levels']['stride%s' % s]
                    rois_num = _rois.shape[0]
                    if rois_num < bucket_size:
                        num_pad = bucket_size - rois_num

                        rois_pad = np.array([[12, 34, 56, 78]] * num_pad)
                        labels_pad = np.array([-1] * num_pad)
                        bbox_targets_pad = np.array([[1, 2, 3, 4] * config.NUM_CLASSES] * num_pad)
                        bbox_weights_pad = np.array([[0, 0, 0, 0] * config.NUM_CLASSES] * num_pad)
                        mask_targets_pad = np.zeros((num_pad, config.NUM_CLASSES, 28, 28),
                                                    dtype=np.int8)
                        mask_weights_pad = np.zeros((num_pad, config.NUM_CLASSES, 1, 1), dtype=np.int8)

                        data_on_imgs['img_%s' % im_i]['rois_on_levels']['stride%s' % s] = np.concatenate(
                            [_rois, rois_pad])
                        data_on_imgs['img_%s' % im_i]['labels_on_levels']['stride%s' % s] = np.concatenate(
                            [_labels, labels_pad])
                        data_on_imgs['img_%s' % im_i]['bbox_targets_on_levels']['stride%s' % s] = np.concatenate(
                            [_bbox_targets, bbox_targets_pad])
                        data_on_imgs['img_%s' % im_i]['bbox_weights_on_levels']['stride%s' % s] = np.concatenate(
                            [_bbox_weights, bbox_weights_pad])
                        data_on_imgs['img_%s' % im_i]['mask_targets_on_levels']['stride%s' % s] = np.concatenate(
                            [_mask_targets, mask_targets_pad])
                        data_on_imgs['img_%s' % im_i]['mask_weights_on_levels']['stride%s' % s] = np.concatenate(
                            [_mask_weights, mask_weights_pad])

            rois_on_imgs = dict()
            labels_on_imgs = dict()
            bbox_targets_on_imgs = dict()
            bbox_weights_on_imgs = dict()
            mask_targets_on_imgs = dict()
            mask_weights_on_imgs = dict()
            for s in config.RCNN_FEAT_STRIDE:
                rois_on_imgs.update({'stride%s' % s: list()})
                labels_on_imgs.update({'stride%s' % s: list()})
                bbox_targets_on_imgs.update({'stride%s' % s: list()})
                bbox_weights_on_imgs.update({'stride%s' % s: list()})
                mask_targets_on_imgs.update({'stride%s' % s: list()})
                mask_weights_on_imgs.update({'stride%s' % s: list()})

            for im_i in range(num_imgs):
                for s in config.RCNN_FEAT_STRIDE:
                    im_rois_on_levels = data_on_imgs['img_%s' % im_i]['rois_on_levels']
                    labels_on_levels = data_on_imgs['img_%s' % im_i]['labels_on_levels']
                    bbox_targets_on_levels = data_on_imgs['img_%s' % im_i]['bbox_targets_on_levels']
                    bbox_weights_on_levels = data_on_imgs['img_%s' % im_i]['bbox_weights_on_levels']
                    mask_targets_on_levels = data_on_imgs['img_%s' % im_i]['mask_targets_on_levels']
                    mask_weights_on_levels = data_on_imgs['img_%s' % im_i]['mask_weights_on_levels']

                    _rois = im_rois_on_levels['stride%s' % s]
                    batch_index = im_i * np.ones((_rois.shape[0], 1))
                    rois_on_imgs['stride%s' % s].append(np.hstack((batch_index, _rois)))
                    labels_on_imgs['stride%s' % s].append(labels_on_levels['stride%s' % s])
                    bbox_targets_on_imgs['stride%s' % s].append(bbox_targets_on_levels['stride%s' % s])
                    bbox_weights_on_imgs['stride%s' % s].append(bbox_weights_on_levels['stride%s' % s])
                    mask_targets_on_imgs['stride%s' % s].append(mask_targets_on_levels['stride%s' % s])
                    mask_weights_on_imgs['stride%s' % s].append(mask_weights_on_levels['stride%s' % s])

            label = dict()
            for s in config.RCNN_FEAT_STRIDE:
                label.update({'label_stride%s' % s: np.reshape(np.concatenate(labels_on_imgs['stride%s' % s], axis=0),
                                                               [num_imgs, -1])})
                label.update({'bbox_target_stride%s' % s: np.reshape(
                    np.concatenate(bbox_targets_on_imgs['stride%s' % s], axis=0), [num_imgs, -1])})
                label.update({'bbox_weight_stride%s' % s: np.reshape(
                    np.concatenate(bbox_weights_on_imgs['stride%s' % s], axis=0), [num_imgs, -1])})
                label.update({'mask_target_stride%s' % s: np.reshape(
                    np.concatenate(mask_targets_on_imgs['stride%s' % s], axis=0),
                    [num_imgs, -1, config.NUM_CLASSES, 28, 28])})
                label.update({'mask_weight_stride%s' % s: np.reshape(
                    np.concatenate(mask_weights_on_imgs['stride%s' % s], axis=0),
                    [num_imgs, -1, config.NUM_CLASSES, 1, 1])})

            # Stack batch data, and update dict
            data = dict()
            data.update({'data': im_array})
            for s in config.RCNN_FEAT_STRIDE:
                rois_array = np.array(rois_on_imgs['stride%s' % s])
                data.update({'rois_stride%s' % s: rois_array})

            data_list.append(data)
            label_list.append(label)

        all_data = dict()
        for key in data_list[0].keys():
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_list[0].keys():
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])

        return all_data, all_label
