.. _TensorFlow-FLuid:

#################
Caffe-Fluid
#################

本文档梳理了Caffe常用Layer与Fluid API对应关系和差异分析。  

接口对应： Caffe Layer与Fluid接口基本一致  

差异对比： Caffe Layer与Fluid接口存在使用或功能差异  

FLuid实现：Fluid无对应接口，但可利用现有接口组合实现  


.. _TensorFlow-FLuid:

#################
TensorFlow-Fluid
#################

.. _a link: http://example.com/

本文档基于TensorFlow v1.12.0梳理了常用API与Fluid API对应关系和差异分析。  

接口对应： TensorFlow与Fluid接口基本一致  

差异对比： 接口存在使用或功能差异  

FLuid实现：Fluid无对应接口，但可利用现有接口组合实现  


..  csv-table:: 
    :header: "序号", "TensorFlow接口", "Fluid接口", "备注"
    :widths: 1, 8, 8, 3

    "1", "`tf.abs <https://www.tensorflow.org/api_docs/python/tf/abs>`_", "`fluid.layers.abs <http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#abs>`_", "接口对应"
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
  
