# edit-mode: -*- python -*-
import paddle.v2 as paddle
#from config.test_conf import cfg
from config.pascal_voc_conf import cfg


def net_conf(mode, scale=1.0):
    """Network configuration. Total three modes included 'train' 'eval'
    and 'infer'. Loss and mAP evaluation layer will return if using 'train'
    and 'eval'. In 'infer' mode, only detection output layer will be returned.
    """
    default_l2regularization = cfg.TRAIN.L2REGULARIZATION

    default_bias_attr = paddle.attr.ParamAttr(l2_rate=0.0, learning_rate=2.0)
    default_static_bias_attr = paddle.attr.ParamAttr(is_static=True)

    def get_param_attr(local_lr, regularization):
        is_static = False
        if local_lr == 0.0:
            is_static = True
        return paddle.attr.ParamAttr(
            learning_rate=local_lr, l2_rate=regularization, is_static=is_static)

    def mbox_block(layer_name, input, num_channels, filter_size, loc_filters,
                   conf_filters):
        #mbox_loc_name = layer_idx + "_mbox_loc"
        mbox_loc = paddle.layer.img_conv(
            #name = layer_name + '_' + 'loc',
            input=input,
            filter_size=filter_size,
            num_channels=num_channels,
            num_filters=loc_filters,
            stride=1,
            padding=0,
            layer_type='exconv',
            bias_attr=default_bias_attr,
            param_attr=get_param_attr(1, default_l2regularization),
            act=paddle.activation.Identity())

        #mbox_conf_name = layer_idx + "_mbox_conf"
        mbox_conf = paddle.layer.img_conv(
            #name = layer_name + '_' + 'conf',
            input=input,
            filter_size=filter_size,
            num_channels=num_channels,
            num_filters=conf_filters,
            stride=1,
            padding=0,
            layer_type='exconv',
            bias_attr=default_bias_attr,
            param_attr=get_param_attr(1, default_l2regularization),
            act=paddle.activation.Identity())

        return mbox_loc, mbox_conf

    def conv_bn_layer(input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      active_type=paddle.activation.Relu(),
                      name=None):
        """
        A wrapper for conv layer with batch normalization layers.
        Note:
        conv layer has no activation.
        """
        tmp = paddle.layer.img_conv(
            #name = name,
            input=input,
            filter_size=filter_size,
            num_channels=channels,
            num_filters=num_filters,
            stride=stride,
            padding=padding,
            groups=num_groups,
            layer_type='exconv',
            # !!! the act in the network with batch norm
            # is paddle.activation.Linear()
            act=active_type,
            # !!! the bias_attr in origin network is False
            bias_attr=True)
        #print tmp.name

        # !!! we have deleted the batch_norm layer here.
        return tmp

    def depthwise_separable(input, num_filters1, num_filters2, num_groups,
                            stride):
        """
        """
        tmp = conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=num_filters1,
            stride=stride,
            padding=1,
            num_groups=num_groups)

        tmp = conv_bn_layer(
            input=tmp,
            filter_size=1,
            num_filters=num_filters2,
            stride=1,
            padding=0)
        return tmp

    img = paddle.layer.data(
        name='image',
        type=paddle.data_type.dense_vector(cfg.IMG_CHANNEL * cfg.IMG_HEIGHT *
                                           cfg.IMG_WIDTH),
        height=cfg.IMG_HEIGHT,
        width=cfg.IMG_WIDTH)

    # conv1: 112x112
    #"conv0"  "conv0/relu"
    conv0 = conv_bn_layer(
        img,
        filter_size=3,
        channels=3,
        num_filters=int(32 * scale),
        stride=2,
        padding=1)

    # 56x56
    # "conv1/dw" "conv1/dw/relu" "conv1" "conv1/relu"
    conv1 = depthwise_separable(
        conv0,
        num_filters1=int(32 * scale),
        num_filters2=int(64 * scale),
        num_groups=int(32 * scale),
        stride=1)

    #"conv2/dw" "conv2/dw/relu" "conv2" "conv2/relu"
    conv2 = depthwise_separable(
        conv1,
        num_filters1=int(64 * scale),
        num_filters2=int(128 * scale),
        num_groups=int(64 * scale),
        stride=2)
    # 28x28
    #"conv3/dw" "conv3/dw/relu" "conv3" "conv3/relu"
    conv3 = depthwise_separable(
        conv2,
        num_filters1=int(128 * scale),
        num_filters2=int(128 * scale),
        num_groups=int(128 * scale),
        stride=1)

    #"conv4/dw" "conv4/dw/relu"  "conv4" "conv4/relu"
    conv4 = depthwise_separable(
        conv3,
        num_filters1=int(128 * scale),
        num_filters2=int(256 * scale),
        num_groups=int(128 * scale),
        stride=2)

    # 14x14
    #"conv5/dw" "conv5/dw/relu" "conv5" "conv5/relu"
    conv5 = depthwise_separable(
        conv4,
        num_filters1=int(256 * scale),
        num_filters2=int(256 * scale),
        num_groups=int(256 * scale),
        stride=1)

    #"conv6/dw" "conv6/dw/relu" "conv6" "conv6/relu"
    conv6 = depthwise_separable(
        conv5,
        num_filters1=int(256 * scale),
        num_filters2=int(512 * scale),
        num_groups=int(256 * scale),
        stride=2)

    tmp = conv6

    # 14x14
    #"conv7/dw" "conv7/dw/relu" "conv7" "conv7/relu"
    #conv7~11
    for i in range(5):
        tmp = depthwise_separable(
            tmp,
            num_filters1=int(512 * scale),
            num_filters2=int(512 * scale),
            num_groups=int(512 * scale),
            stride=1)
    conv11 = tmp

    # 7x7
    #"conv12/dw" "conv12/dw/relu" "conv12" "conv12/relu"
    conv12 = depthwise_separable(
        conv11,
        num_filters1=int(512 * scale),
        num_filters2=int(1024 * scale),
        num_groups=int(512 * scale),
        stride=2)

    #"conv13/dw" "conv13/dw/relu" "conv13" "conv13/relu"
    conv13 = depthwise_separable(
        conv12,
        num_filters1=int(1024 * scale),
        num_filters2=int(1024 * scale),
        num_groups=int(1024 * scale),
        stride=1)

    # add begin
    # conv14_1 "conv14_1/relu"
    conv14_1 = conv_bn_layer(
        #name = 'module3_1',
        input=conv13,
        filter_size=1,
        num_filters=int(256 * scale),
        stride=1,
        padding=0)

    #conv14_2 "conv14_2/relu"
    conv14_2 = conv_bn_layer(
        #name = 'module3_2',
        input=conv14_1,
        filter_size=3,
        num_filters=int(512 * scale),
        stride=2,
        padding=1)

    #conv15_1 "conv15_1/relu"
    conv15_1 = conv_bn_layer(
        # name = 'module4_1',
        input=conv14_2,
        filter_size=1,
        num_filters=int(128 * scale),
        stride=1,
        padding=0)

    #"conv15_2"  "conv15_2/relu"
    conv15_2 = conv_bn_layer(
        #name = 'module4_2',
        input=conv15_1,
        filter_size=3,
        num_filters=int(256 * scale),
        stride=2,
        padding=1)

    #conv16_1 "conv16_1/relu"
    conv16_1 = conv_bn_layer(
        #name = 'module5_1',
        input=conv15_2,
        filter_size=1,
        num_filters=int(128 * scale),
        stride=1,
        padding=0)

    #"conv16_2"  "conv16_2/relu"
    conv16_2 = conv_bn_layer(
        #name = 'module5_2',
        input=conv16_1,
        filter_size=3,
        num_filters=int(256 * scale),
        stride=2,
        padding=1)

    #conv17_1 conv17_1/relu
    conv17_1 = conv_bn_layer(
        #name = 'module6_1',
        input=conv16_2,
        filter_size=1,
        num_filters=int(64 * scale),
        stride=1,
        padding=0)

    #conv17_2 conv17_2/relu
    conv17_2 = conv_bn_layer(
        #name = 'module6_2',
        input=conv17_1,
        filter_size=3,
        num_filters=int(128 * scale),
        stride=2,
        padding=1)

    conv11_mbox_priorbox = paddle.layer.priorbox(
        input=conv11,
        image=img,
        min_size=cfg.NET.CONV11.PB.MIN_SIZE,
        aspect_ratio=cfg.NET.CONV11.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV11.PB.VARIANCE)

    conv11_norm = paddle.layer.cross_channel_norm(
        name="conv11_norm",
        input=conv11,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))

    conv11_mbox_loc, conv11_mbox_conf= \
        mbox_block("module1", conv11_norm, int(512*scale), 1, 12, 63) # kernel_size=1

    conv13_mbox_priorbox = paddle.layer.priorbox(
        input=conv13,
        image=img,
        min_size=cfg.NET.CONV13.PB.MIN_SIZE,
        max_size=cfg.NET.CONV13.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.CONV13.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV13.PB.VARIANCE)
    conv13_norm = paddle.layer.cross_channel_norm(
        name="conv13_norm",
        input=conv13,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))
    conv13_mbox_loc, conv13_mbox_conf= \
        mbox_block("module2", conv13_norm, int(1024*scale), 1, 24, 126)

    conv14_2_mbox_priorbox = paddle.layer.priorbox(
        input=conv14_2,
        image=img,
        min_size=cfg.NET.CONV14_2.PB.MIN_SIZE,
        max_size=cfg.NET.CONV14_2.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.CONV14_2.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV14_2.PB.VARIANCE)
    conv14_2_norm = paddle.layer.cross_channel_norm(
        name="conv14_2",
        input=conv14_2,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))
    conv14_2_mbox_loc, conv14_2_mbox_conf= \
            mbox_block("module3", conv14_2_norm, int(512*scale), 1, 24, 126)

    conv15_2_mbox_priorbox = paddle.layer.priorbox(
        input=conv15_2,
        image=img,
        min_size=cfg.NET.CONV15_2.PB.MIN_SIZE,
        max_size=cfg.NET.CONV15_2.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.CONV15_2.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV15_2.PB.VARIANCE)
    conv15_2_norm = paddle.layer.cross_channel_norm(
        name="conv15_2_norm",
        input=conv15_2,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))

    conv15_2_mbox_loc, conv15_2_mbox_conf= \
            mbox_block("module4", conv15_2_norm, int(256*scale), 1, 24, 126)

    conv16_2_mbox_priorbox = paddle.layer.priorbox(
        input=conv16_2,
        image=img,
        min_size=cfg.NET.CONV16_2.PB.MIN_SIZE,
        max_size=cfg.NET.CONV16_2.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.CONV16_2.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV16_2.PB.VARIANCE)
    conv16_2_norm = paddle.layer.cross_channel_norm(
        name="conv16_2_norm",
        input=conv16_2,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))
    conv16_2_mbox_loc, conv16_2_mbox_conf= \
        mbox_block("module5", conv16_2_norm, int(256*scale), 1, 24, 126)

    conv17_2_mbox_priorbox = paddle.layer.priorbox(
        input=conv17_2,
        image=img,
        min_size=cfg.NET.CONV17_2.PB.MIN_SIZE,
        max_size=cfg.NET.CONV17_2.PB.MAX_SIZE,
        aspect_ratio=cfg.NET.CONV17_2.PB.ASPECT_RATIO,
        variance=cfg.NET.CONV17_2.PB.VARIANCE)
    conv17_2_norm = paddle.layer.cross_channel_norm(
        name="conv17_2_norm",
        input=conv17_2,
        param_attr=paddle.attr.ParamAttr(
            initial_mean=20, initial_std=0, is_static=False, learning_rate=1))
    conv17_2_mbox_loc, conv17_2_mbox_conf= \
        mbox_block("module6", conv17_2_norm, int(128*scale), 1, 24, 126)

    mbox_priorbox = paddle.layer.concat(
        name="mbox_priorbox",
        input=[
            conv11_mbox_priorbox, conv13_mbox_priorbox, conv14_2_mbox_priorbox,
            conv15_2_mbox_priorbox, conv16_2_mbox_priorbox,
            conv17_2_mbox_priorbox
        ])

    loc_loss_input = [
        conv11_mbox_loc, conv13_mbox_loc, conv14_2_mbox_loc, conv15_2_mbox_loc,
        conv16_2_mbox_loc, conv17_2_mbox_loc
    ]

    conf_loss_input = [
        conv11_mbox_conf, conv13_mbox_conf, conv14_2_mbox_conf,
        conv15_2_mbox_conf, conv16_2_mbox_conf, conv17_2_mbox_conf
    ]

    detection_out = paddle.layer.detection_output(
        input_loc=loc_loss_input,
        input_conf=conf_loss_input,
        priorbox=mbox_priorbox,
        confidence_threshold=cfg.NET.DETOUT.CONFIDENCE_THRESHOLD,
        nms_threshold=cfg.NET.DETOUT.NMS_THRESHOLD,
        num_classes=cfg.CLASS_NUM,
        nms_top_k=cfg.NET.DETOUT.NMS_TOP_K,
        keep_top_k=cfg.NET.DETOUT.KEEP_TOP_K,
        background_id=cfg.BACKGROUND_ID,
        name="detection_output")

    if mode == 'train' or mode == 'eval':
        bbox = paddle.layer.data(
            name='bbox', type=paddle.data_type.dense_vector_sequence(6))
        loss = paddle.layer.multibox_loss(
            input_loc=loc_loss_input,
            input_conf=conf_loss_input,
            priorbox=mbox_priorbox,
            label=bbox,
            num_classes=cfg.CLASS_NUM,
            overlap_threshold=cfg.NET.MBLOSS.OVERLAP_THRESHOLD,
            neg_pos_ratio=cfg.NET.MBLOSS.NEG_POS_RATIO,
            neg_overlap=cfg.NET.MBLOSS.NEG_OVERLAP,
            background_id=cfg.BACKGROUND_ID,
            name="multibox_loss")
        paddle.evaluator.detection_map(
            input=detection_out,
            label=bbox,
            overlap_threshold=cfg.NET.DETMAP.OVERLAP_THRESHOLD,
            background_id=cfg.BACKGROUND_ID,
            evaluate_difficult=cfg.NET.DETMAP.EVAL_DIFFICULT,
            ap_type=cfg.NET.DETMAP.AP_TYPE,
            name="detection_evaluator")
        return loss, detection_out
    elif mode == 'infer':
        return detection_out


if __name__ == '__main__':
    out = net_conf('infer', scale=1.0)
