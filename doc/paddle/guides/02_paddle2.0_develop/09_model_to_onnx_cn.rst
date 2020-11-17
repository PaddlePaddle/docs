
#############
模型导出ONNX协议
#############

一、ONNX简介
##################

ONNX (Open Neural Network Exchange) 是一种针对机器学习所设计的开源文件格式，用于存储训练好的模型，它使得不同的人工智能框架可以采用相同格式存储模型并交互，通过 ``ONNX`` 格式，Paddle 模型可以使用OpenVINO、ONNX Runtime等框架进行推理。下面介绍如何将训练好的 ResNet50 v1.5 模型转换为 ONNX 模型并验证正确性。

二、将模型导出为ONNX协议 
##################

2.1 动态图模型
------------

.. code-block:: python

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        def forward(self, x):
            return self._linear(x)


2.2 导出ONNX协议
------------

Paddle模型转换为ONNX协议，首先会将Paddle的动态图``paddle.nn.Layer``转换为静态图，即全部由算子组成的网络结构，然后依照ONNX的算子协议，将Paddle的算子一一映射为ONNX的算子。动态图转换为ONNX调用 ``paddle.onnx.export()`` 接口即可实现，通过 ``input_spec`` 参数为模型指定输入的形状和数据类型，支持 ``Tensor`` 或 ``InputSpec`` ，其中 ``InputSpec`` 支持动态的shape。

关于``paddle.onnx.export`` 接口更详细的使用方法，请参考API。

.. code-block:: python

    from paddle.static import InputSpec
    save_path = 'onnx.save/linear_net'
    input = InputSpec([None, 784], 'float32', 'x')
    paddle.onnx.export(layer, save_path, input_spec=[input])

2.3 静态图转ONNX协议
------------

Paddle 2.0以后我们将主推动态图组网方式，如果您的模型是静态图，请参考paddle2onnx的使用文档。

三、ONNX模型的验证
##################

这里介绍两种方法来检查转换出来ONNX模型的正确性：
ONNX模型的验证主要包括两个方面，一个是算子是否符合对应版本的协议，第二个是网络结构是否完整，

3.1 ONNX Checker

.. code-block:: python

    import onnx
    onnx_model = onnx.load(save_path + '.onnx')
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

如果模型检查失败，请到 ` Paddle  <https://github.com/PaddlePaddle/Paddle/issues/>`_ 或 ` paddle2onnx  <https://github.com/PaddlePaddle/paddle2onnx/issues/>`_ 提出Issue，我们会跟进相应的问题。

3.2 Netorn可视化
------------
当转换出来的 ``ONNX`` 模型在推理的过程中报错，首先根据报错的信息确认是推理引擎的局限性，还是模型的问题，
如果模型的问题可以

四、使用onnxruntime推理ONNX模型 
##################

.. code-block:: python

	import onnxruntime

	ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
	
	def to_numpy(tensor):
	    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
	
	# compute ONNX Runtime output prediction
	ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
	ort_outs = ort_session.run(None, ort_inputs)
	
	# compare ONNX Runtime and PyTorch results
	np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
	
	print("Exported model has been tested with ONNXRuntime, and the result looks good!")

五、算子转换支持列表
##################

| Paddle kernel |  ONNX Opset Versions | support_status |
| ------------- | ------------------------------- | -----------------|
| abs | 1~12 |
| arg_max | 1~12 |
| assign_value | 1~12 |
| batch_norm | 1~12 |
| bilinear_interp | 9~12 |
| box_coder | 7~12 |
| cast | 1~12 |
| clip | 1~12 |
| concat | 1~12 |
| conv2d | 1~12 |
| conv2d_transpose | 1~12 |
| depthwise_conv2d | 1~12 |
| dropout | 7~12 |
| elementwise_add | 7 ~ 12 |
| elementwise_sub | 7 ~ 12 |
| elementwise_mul | 7 ~ 12 |
| elementwise_div | 7 ~ 12 |
| exp | 1~12 |
| fill_constant | 1~12 |
| fill_any_like | 9~12 |
| flatten2 | 1~12 |
| floor | 1~12 |
| gather | 1~12 |  opset 1~10 limited supported |
| hard_sigmoid | 1~12 |
| hard_swish | 1~12 |
| im2sequence | 1~12 |
| instance_norm | 1~12 |
| leaky_relu | 1~12 |
| log | 1~12 |
| matmul | 1~12 |
| mul | 1~12 |
| muticlass_nms | 10~12 |
| muticlass_nms2 | 10~12 |
| nearest_interp | 9~12 |
| norm | 1~12 |
| pad2d | 1~12 |
| pool2d | 1~12 | limited supported |
| pow | 8~12 |
| prior_box | 1~12 |
| prelu | 1~12 |
| reciprocal | 1~12 |
| reduce_mean | 1~12 |
| reduce_max | 1~12 |
| reduce_min | 1~12 |
| reduce_sum | 1~12 |
| relu | 1~12 |
| relu6 | 1~12 |
| reshape2 | 5~12 |
| roi_align | 10~12 |
| softmax | 1~12 |
| scale | 1~12 | opset 1~6 limited supported |
| shape | 1~12 |
| sigmoid | 1~12 |
| slice | 1~12 |
| split | 1~12 |
| sum | 1~12 |
| squeeze2 | 1~12 |
| swish | 1~12 |
| tanh | 1~12 |
| transpose2 | 1~12 |
| uniform_random_batch_size_like | 1~12 |
| unsqueeze2 | 1~12 |
| yolo_box | 9~12 |


六、模型转换支持列表
##################
