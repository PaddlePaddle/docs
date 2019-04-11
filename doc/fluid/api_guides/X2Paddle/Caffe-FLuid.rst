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
 
