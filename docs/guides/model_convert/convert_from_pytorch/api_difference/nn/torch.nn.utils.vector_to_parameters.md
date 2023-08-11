## [ 参数完全一致 ]torch.nn.utils.vector_to_parameters

### [torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/stable/generated/torch.nn.utils.vector_to_parameters.html#torch-nn-utils-vector-to-parameters)

```python
torch.nn.utils.vector_to_parameters(vec, parameters)
```

### [paddle.nn.utils.vector_to_parameters](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/utils/vector_to_parameters_cn.html#vector-to-parameters)

```python
paddle.nn.utils.vector_to_parameters(vec, parameters, name=None)
```

两者功能一致，参数用法一致，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| vec   | vec  | 一个 1-D Tensor，它将被切片并复制到输入参数(input parameters)中。                                            |
| parameters   | parameters  | 可迭代的多个 parameter。                                            |
