.. _TensorFlow-FLuid:

#################
Caffe-Fluid
#################

本文档梳理了Caffe常用Layer与Fluid API对应关系和差异分析。  

接口对应： Caffe Layer与Fluid接口基本一致  

差异对比： Caffe Layer与Fluid接口存在使用或功能差异  

FLuid实现：Fluid无对应接口，但可利用现有接口组合实现  

..  csv-table:: 
    :header: "序号", "TensorFlow接口", "Fluid接口", "备注"
    :widths: 1, 8, 8, 3
    
  "1", "`AbsVal <http://caffe.berkeleyvision.org/tutorial/layers/absval.html>`_", "`fluid.layers.abs <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-182-abs>`_", "接口对应"
  "2", "`Accuracy <http://caffe.berkeleyvision.org/tutorial/layers/accuracy.html>`_", "`fluid.layers.accuracy <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-253-accuracy>`_", "`差异对比 <Accuracy.md>`_"
  "3", "`ArgMax <http://caffe.berkeleyvision.org/tutorial/layers/argmax.html>`_", "`fluid.layers.argmax <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-204-argmax>`_", "`差异对比 <ArgMax.md>`_"
  "4", "`BatchNorm <http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html>`_", "`fluid.layers.batch_norm <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-36-batch_norm>`_", "`差异对比 <BatchNorm.md>`_"
  "5", "`BNLL <http://caffe.berkeleyvision.org/tutorial/layers/bnll.html>`_", "`fluid.layers.softplus <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-194-softplus>`_", "接口对应"
  "6", "`Concat <http://caffe.berkeleyvision.org/tutorial/layers/concat.html>`_", "`fluid.layers.concat <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-209-concat>`_", "接口对应"
  "7", "`Convolution <http://caffe.berkeleyvision.org/tutorial/layers/convolution.html>`_", "`fluid.layers.conv2d <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-45-conv2d>`_", "`差异对比 <Convolution.md>`_"
  "8", "`Crop <http://caffe.berkeleyvision.org/tutorial/layers/crop.html>`_", "`fluid.layers.crop <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-51-crop>`_", "`差异对比 <Crop.md>`_"
  "9", "`Deconvolution <http://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html>`_", "`fluid.layers.conv2d_transpose <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-46-conv2d_transpose>`_", "`差异对比 <Deconvolution.md>`_"
  "10", "`Dropout <http://caffe.berkeleyvision.org/tutorial/layers/dropout.html>`_", "`fluid.layers.dropout <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-56-dropout>`_", "`差异对比 <Dropout.md>`_"
  "11", "`Eltwise <http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html>`_", "-", "`Fluid实现 <Eltwise.md>`_"
  "12", "`ELU <http://caffe.berkeleyvision.org/tutorial/layers/elu.html>`_", "`fluid.layers.elu <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-68-elu>`_", "接口对应"
  "13", "`EuclideanLoss <http://caffe.berkeleyvision.org/tutorial/layers/euclideanloss.html>`_", "`fluid.layers.square_error_cost <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-167-square_error_cost>`_", "`差异对比 <EuclideanLoss.md>`_"
  "14", "`Exp <http://caffe.berkeleyvision.org/tutorial/layers/exp.html>`_", "`fluid.layers.exp <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-186-exp>`_", "`差异对比 <Exp.md>`_"
  "15", "`Flatten <http://caffe.berkeleyvision.org/tutorial/layers/flatten.html>`_", "`fluid.layers.reshape <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-72-reshape>`_", "`差异对比 <Flatten.md>`_"
  "16", "`InnerProduct <http://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html>`_", "`fluid.layers.fc <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-71-fc>`_", "`差异对比 <InnerProduct.md>`_"
  "17", "`Input <http://caffe.berkeleyvision.org/tutorial/layers/input.html>`_", "`fluid.layers.data <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-20-data>`_", "`差异对比 <Input.md>`_"
  "18", "`Log <http://caffe.berkeleyvision.org/tutorial/layers/log.html>`_", "`fluid.layers.log <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-93-log>`_", "`差异对比 <Log.md>`_"
  "19", "`LRN <http://caffe.berkeleyvision.org/tutorial/layers/lrn.html>`_", "`fluid.layers.lrn <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-99-lrn>`_", "`差异对比 <LRN.md>`_"
  "20", "`Pooling <http://caffe.berkeleyvision.org/tutorial/layers/pooling.html>`_", "`fluid.layers.pool2d <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-115-pool2d>`_", "`差异对比 <Pooling.md>`_"
  "21", "`Power <http://caffe.berkeleyvision.org/tutorial/layers/power.html>`_", "`fluid.layers.pow <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-117-pow>`_", "`差异对比 <Power.md>`_"
  "22", "`PReLU <http://caffe.berkeleyvision.org/tutorial/layers/prelu.html>`_", "`fluid.layers.prelu <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-118-prelu>`_", "接口对应"
  "23", "`Reduction <http://caffe.berkeleyvision.org/tutorial/layers/reduction.html>`_", "-", "`Fluid实现 <Reduction.md>`_"
  "24", "`ReLU <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`_", "`fluid.layers.leaky_relu <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-128-relu>`_", "接口对应"
  "25", "`Reshape <http://caffe.berkeleyvision.org/tutorial/layers/reshape.html>`_", "`fluid.layers.reshape <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-130-reshape>`_", "`差异对比 <Reshape.md>`_"
  "26", "`SigmoidCrossEntropyLoss <http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html>`_", "`fluid.layers.sigmoid_cross_entropy_with_logits <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-158-sigmoid_cross_entropy_with_logits>`_", "`差异对比 <SigmoidCrossEntropyLoss.md>`_"
  "27", "`Sigmoid <http://caffe.berkeleyvision.org/tutorial/layers/sigmoid.html>`_", "`fluid.layers.sigmoid <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-192-sigmoid>`_", "接口对应"
  "28", "`Slice <http://caffe.berkeleyvision.org/tutorial/layers/slice.html>`_", "`fluid.layers.slice <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-160-slice>`_", "`差异对比 <Slice.md>`_"
  "29", "`SoftmaxWithLoss <http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html>`_", "`fluid.layers.softmax_with_cross_entropy <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-164-softmax_with_cross_entropy>`_", "`差异对比 <SofmaxWithLoss.md>`_"
  "30", "`Softmax <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`_", "`fluid.layers.softmax <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-163-softmax>`_", "`差异对比 <Sofmax.md>`_"
  "31", "`TanH <http://caffe.berkeleyvision.org/tutorial/layers/tanh.html>`_", "`fluid.layers.tanh <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-199-tanh>`_", "接口对应"
  "32", "`Tile <http://caffe.berkeleyvision.org/tutorial/layers/tile.html>`_", "`fluid.layers.expand <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-70-expand>`_", "`差异对比 <Tile.md>`_"
