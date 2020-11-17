
#############
模型导出ONNX协议
#############

一、简介
##################

ONNX (Open Neural Network Exchange) 是针对机器学习所设计的开源文件格式，用于存储训练好的模型，它使得不同的人工智能框架可以采用相同格式存储模型并交互，通过ONNX格式，Paddle 模型可以使用OpenVINO、ONNX Runtime等框架进行推理。Paddle转ONNX协议由 `paddle2onnx <https://github.com/PaddlePaddle/paddle2onnx>`_ 实现，可用于图像分类、检测、分割等类型的转换，同时也在持续探索NLP类模型的转换。下面介绍如何将Paddle模型转换为ONNX模型并验证正确性。

本教程涉及的 `示例代码 <https://github.com/paddlepaddle/paddle2onnx/blob/develop/examples/tutorial_dygraph2onnx.py>`_ ， 除Paddle以外，还需安装以下依赖：

.. code-block:: bash

    pip install paddle2onnx onnx onnxruntime 

二、模型导出为ONNX协议 
##################

2.1 动态图导出ONNX协议
------------

Paddle动态图模型转换为ONNX协议，首先会将Paddle的动态图 ``paddle.nn.Layer`` 转换为静态图， 详细原理可以参考 `动态图转静态图 <../04_dygraph_to_static/index_cn.html>`_ ，然后依照ONNX的算子协议，将Paddle的算子一一映射为ONNX的算子。动态图转换ONNX调用 ``paddle.onnx.export()`` 接口即可实现，该接口通过 ``input_spec`` 参数为模型指定输入的形状和数据类型，支持 ``Tensor`` 或 ``InputSpec`` ，其中 ``InputSpec`` 支持动态的shape。

关于 ``paddle.onnx.export`` 接口更详细的使用方法，请参考 `API <../../api/paddle/onnx/export_cn.rst>`_ 。

.. code-block:: python

    import paddle
    from paddle.static import InputSpec

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(784, 10)

        def forward(self, x):
            return self._linear(x)

    # export to ONNX 
    layer = LinearNet()
    save_path = 'onnx.save/linear_net'
    x_spec = InputSpec([None, 784], 'float32', 'x')
    paddle.onnx.export(layer, save_path, input_spec=[x_spec])

2.2 静态图导出ONNX协议
------------

Paddle 2.0以后我们将主推动态图组网方式，如果您的模型来自于旧版本的Paddle，请参考paddle2onnx的 `使用文档 <https://github.com/PaddlePaddle/paddle2onnx/blob/develop/README.md>`_ 和 `示例 <https://github.com/paddlepaddle/paddle2onnx/blob/develop/examples/tutorial.ipynb>`_ 。

三、ONNX模型的验证
##################

ONNX官方工具包提供了API可验证模型的正确性，主要包括两个方面，一是算子是否符合对应版本的协议，二是网络结构是否完整。

.. code-block:: python

    # check by ONNX
    import onnx

    onnx_file = save_path +  '.onnx'
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

如果模型检查失败，请到 `Paddle  <https://github.com/PaddlePaddle/Paddle/issues/>`_ 或 `paddle2onnx  <https://github.com/PaddlePaddle/paddle2onnx/issues/>`_ 提出Issue，我们会跟进相应的问题。

四、ONNXRuntime推理
##################

.. code-block:: python

    import numpy as np
    import onnxruntime
    
    x = np.random.random((2, 784)).astype('float32')
    
    # predict by ONNX Runtime
    ort_sess = onnxruntime.InferenceSession(onnx_file)
    ort_inputs = {ort_sess.get_inputs()[0].name: x}
    ort_outs = ort_sess.run(None, ort_inputs)
    
    print("Exported model has been predict by ONNXRuntime!") 
    
    # predict by Paddle
    layer.eval() 
    paddle_outs = layer(x)
    
    # compare ONNX Runtime and Paddle results
    np.testing.assert_allclose(ort_outs[0], paddle_outs.numpy(), rtol=1.0, atol=1e-05)
    
    print("The difference of result between ONNXRuntime and Paddle looks good!")


五、相关链接
##################

 - `算子转换支持列表  <https://github.com/paddlepaddle/paddle2onnx/blob/develop/docs/op_list.md>`_ 
 - `模型转换支持列表 <https://github.com/PaddlePaddle/paddle2onnx/blob/develop/docs/model_zoo.md>`_ 
