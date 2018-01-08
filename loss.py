# coding:utf-8

from mxnet import gluon
from utils import *


class TotalLoss(gluon.loss.Loss):
    def __init__(self, s, c, b, scale_coordinate=5, scale_noobject_conf=0.5, scale_object_conf=1, scale_class_prob=1,
                 batch_axis=0, axis=-1, **kwargs):
        super(TotalLoss, self).__init__(None, batch_axis, **kwargs)
        self.s = s
        self._axis = axis
        self.c = c
        self.b = b
        self._scale_class_prob = scale_class_prob
        self._scale_object_conf = scale_object_conf
        self._scale_noobject_conf = scale_noobject_conf
        self._scale_coordinate = scale_coordinate
        self._class_error_func = gluon.loss.SoftmaxCrossEntropyLoss()


    def _split_y(self, y ):
        """
        split y as labels, preds, location
        :param y:
        :param s:
        :param b:
        :param c:
        :return: labels_, preds_, location_
        """
        label, preds, location = deal_output(y, self.s, self.b, self.c )
        return label, preds, location


    def _iou2(self, box, box_label):
        """

        :param box:
        :param box_label:
        :return:
        """
        wh = box[:, :, :, 2:4]
        wh = nd.power(wh, 2)
        center = box[:, :, :, 0:1]
        predict_areas = wh[:, :, :, 0] * wh[:, :, :, 1]

        predict_bottom_right = center + 0.5 * wh
        predict_top_left = center - 0.5 * wh

        wh = box_label[:, :, :, 2:4]
        wh = nd.power(wh, 2)
        center = box_label[:, :, :, 0:1]
        label_areas = wh[:, :, :, 0] * wh[:, :, :, 1]

        label_bottom_right = center + 0.5 * wh
        label_top_left = center - 0.5 * wh

        temp = nd.concat(*[predict_top_left[:, :, :, 0:1], label_top_left[:, :, :, 0:1]], dim=3)

        temp_max1 = nd.max(temp, axis=3)
        temp_max1 = nd.expand_dims(temp_max1, axis=3)
        temp = nd.concat(*[predict_top_left[:, :, :, 1:], label_top_left[:, :, :, 1:]], dim=3)
        temp_max2 = nd.max(temp, axis=3)
        temp_max2 = nd.expand_dims(temp_max2, axis=3)

        intersect_top_left = nd.concat(*[temp_max1, temp_max2], dim=3)
        temp = nd.concat(*[predict_bottom_right[:, :, :, 0:1], label_bottom_right[:, :, :, 0:1]], dim=3)
        temp_min1 = nd.min(temp, axis=3)
        temp_min1 = nd.expand_dims(temp_min1, axis=3)
        temp = nd.concat(*[predict_bottom_right[:, :, :, 1:], label_bottom_right[:, :, :, 1:]], dim=3)
        temp_min2 = nd.min(temp, axis=3)
        temp_min2 = nd.expand_dims(temp_min2, axis=3)

        intersect_bottom_right = nd.concat(*[temp_min1, temp_min2], dim=3)
        intersect_wh = intersect_bottom_right - intersect_top_left
        intersect_wh = nd.relu(intersect_wh)
        intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]
        ious = intersect / (predict_areas + label_areas - intersect)
        # print(nd.max(iou,2).shape)
        max_iou = nd.expand_dims(nd.max(ious,2),axis=2)
        best_ = nd.equal(max_iou,ious)
        best_boat = nd.ones(shape = ious.shape)
        #best_boat = best_.copy()
        for batch in range(len(best_)):
             best_boat[batch] = best_[batch]
        #for iou
        return nd.reshape(best_boat, shape=(-1, self.s * self.s * self.b))

    def _calculate_preds_loss(self, label, local_pre, local_label):
        """
        :param ypre:
        :param label:
        :return:
        """
        ious = self._iou2(local_pre, local_label)
        confident = label * ious
        return confident

    def hybrid_forward(self, F, ypre, label):
        assert ypre.shape == label.shape, "Fuck "
        label_pre, preds_pre, location_pre = self._split_y(ypre)
        label_real, preds_real, location_real = self._split_y(label)
        batch_size = len(label_real)
        loss = nd.square(ypre - label)
        class_weight = nd.ones(
        shape = (batch_size, self.s*self.s*self.c)) *self._scale_class_prob
        location_weight = nd.ones(
        shape = (batch_size, self.s * self.s * self.b, 4))
        confs = self._calculate_preds_loss(preds_real, location_pre, location_real)
        preds_weight = self._scale_noobject_conf * (
        1. - confs) + self._scale_object_conf * confs  # self.s * self.s * self.b
        location_weight = (nd.expand_dims(preds_weight, axis=2) * location_weight) * self._scale_coordinate
        location_weight = nd.reshape(location_weight, (-1, self.s * self.s * self.b * 4))
        W = nd.concat(*[class_weight, preds_weight, location_weight], dim=1)
        total_loss = nd.sum(loss * W, 1)
        return total_loss
