.. _Caffe-Fluid:

########################
Caffe-Fluid 常用层对应表
########################

本文档梳理了 Caffe 常用 Layer 与 PaddlePaddle API 对应关系和差异分析。根据文档对应关系，有 Caffe 使用经验的用户，可根据对应关系，快速熟悉 PaddlePaddle 的接口使用。


..  csv-table::
    :header: "序号", "Caffe Layer", "Fluid 接口", "备注"
    :widths: 1, 8, 8, 3

    "1",  "`AbsVal <http://caffe.berkeleyvision.org/tutorial/layers/absval.html>`_", ":ref:`cn_api_fluid_layers_abs`",  "功能一致"
    "2",  "`Accuracy <http://caffe.berkeleyvision.org/tutorial/layers/accuracy.html>`_", ":ref:`cn_api_fluid_layers_accuracy`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Accuracy.md>`_"
    "3",  "`ArgMax <http://caffe.berkeleyvision.org/tutorial/layers/argmax.html>`_", ":ref:`cn_api_fluid_layers_argmax`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/ArgMax.md>`_"
    "4",  "`BatchNorm <http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html>`_", ":ref:`cn_api_fluid_layers_batch_norm`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/BatchNorm.md>`_"
    "5",  "`BNLL <http://caffe.berkeleyvision.org/tutorial/layers/bnll.html>`_", ":ref:`cn_api_fluid_layers_softplus`",  "功能一致"
    "6",  "`Concat <http://caffe.berkeleyvision.org/tutorial/layers/concat.html>`_", ":ref:`cn_api_fluid_layers_concat`",  "功能一致"
    "7",  "`Convolution <http://caffe.berkeleyvision.org/tutorial/layers/convolution.html>`_", ":ref:`cn_api_fluid_layers_conv2d`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Convolution.md>`_"
    "8",  "`Crop <http://caffe.berkeleyvision.org/tutorial/layers/crop.html>`_", ":ref:`cn_api_fluid_layers_crop`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Crop.md>`_"
    "9",  "`Deconvolution <http://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html>`_", ":ref:`cn_api_fluid_layers_conv2d_transpose`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Deconvolution.md>`_"
    "10",  "`Dropout <http://caffe.berkeleyvision.org/tutorial/layers/dropout.html>`_", ":ref:`cn_api_fluid_layers_dropout`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Dropout.md>`_"
    "11",  "`Eltwise <http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html>`_",  "无相应接口",  "`Fluid 实现 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Eltwise.md>`_"
    "12",  "`ELU <http://caffe.berkeleyvision.org/tutorial/layers/elu.html>`_", ":ref:`cn_api_fluid_layers_elu`",  "功能一致"
    "13",  "`EuclideanLoss <http://caffe.berkeleyvision.org/tutorial/layers/euclideanloss.html>`_", ":ref:`cn_api_fluid_layers_square_error_cost`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/EuclideanLoss.md>`_"
    "14",  "`Exp <http://caffe.berkeleyvision.org/tutorial/layers/exp.html>`_", ":ref:`cn_api_fluid_layers_exp`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Exp.md>`_"
    "15",  "`Flatten <http://caffe.berkeleyvision.org/tutorial/layers/flatten.html>`_", ":ref:`cn_api_fluid_layers_reshape`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Flatten.md>`_"
    "16",  "`InnerProduct <http://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html>`_", ":ref:`cn_api_fluid_layers_fc`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/InnerProduct.md>`_"
    "17",  "`Input <http://caffe.berkeleyvision.org/tutorial/layers/input.html>`_", ":ref:`cn_api_fluid_layers_data`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Input.md>`_"
    "18",  "`Log <http://caffe.berkeleyvision.org/tutorial/layers/log.html>`_", ":ref:`cn_api_fluid_layers_log`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Log.md>`_"
    "19",  "`LRN <http://caffe.berkeleyvision.org/tutorial/layers/lrn.html>`_", ":ref:`cn_api_fluid_layers_lrn`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/LRN.md>`_"
    "20",  "`Pooling <http://caffe.berkeleyvision.org/tutorial/layers/pooling.html>`_", ":ref:`cn_api_fluid_layers_pool2d`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Pooling.md>`_"
    "21",  "`Power <http://caffe.berkeleyvision.org/tutorial/layers/power.html>`_", ":ref:`cn_api_fluid_layers_pow`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Power.md>`_"
    "22",  "`PReLU <http://caffe.berkeleyvision.org/tutorial/layers/prelu.html>`_", ":ref:`cn_api_fluid_layers_prelu`",  "功能一致"
    "23",  "`Reduction <http://caffe.berkeleyvision.org/tutorial/layers/reduction.html>`_",  "无相应接口",  "`Fluid 实现 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Reduction.md>`_"
    "24",  "`ReLU <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`_", ":ref:`cn_api_fluid_layers_leaky_relu`",  "功能一致"
    "25",  "`Reshape <http://caffe.berkeleyvision.org/tutorial/layers/reshape.html>`_", ":ref:`cn_api_fluid_layers_reshape`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Reshape.md>`_"
    "26",  "`SigmoidCrossEntropyLoss <http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html>`_", ":ref:`cn_api_fluid_layers_sigmoid_cross_entropy_with_logits`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/SigmoidCrossEntropyLoss.md>`_"
    "27",  "`Sigmoid <http://caffe.berkeleyvision.org/tutorial/layers/sigmoid.html>`_", ":ref:`cn_api_fluid_layers_sigmoid`",  "功能一致"
    "28",  "`Slice <http://caffe.berkeleyvision.org/tutorial/layers/slice.html>`_", ":ref:`cn_api_fluid_layers_slice`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Slice.md>`_"
    "29",  "`SoftmaxWithLoss <http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html>`_", ":ref:`cn_api_fluid_layers_softmax_with_cross_entropy`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/SofmaxWithLoss.md>`_"
    "30",  "`Softmax <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`_", ":ref:`cn_api_fluid_layers_softmax`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Sofmax.md>`_"
    "31",  "`TanH <http://caffe.berkeleyvision.org/tutorial/layers/tanh.html>`_", ":ref:`cn_api_fluid_layers_tanh`",  "功能一致"
    "32",  "`Tile <http://caffe.berkeleyvision.org/tutorial/layers/tile.html>`_", ":ref:`cn_api_fluid_layers_expand`",  "`差异对比 <https://github.com/PaddlePaddle/X2Paddle/blob/master/caffe2fluid/doc/Tile.md>`_"
