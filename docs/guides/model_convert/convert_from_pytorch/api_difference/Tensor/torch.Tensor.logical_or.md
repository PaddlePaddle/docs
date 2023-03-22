## [ 仅 paddle 参数更多 ] torch.Tensor.logical_or

### [torch.Tensor.logical_or](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_or.html)

```python
torch.Tensor.logical_or()
```

### [paddle.Tensor.logical_or](https://www.paddlepaddle.org.cn/documentation/docs/api/paddle/Tensor_cn.html#logical-or-y-out-none-name-none)

```python
paddle.Tensor.logical_or(y, 
                         out=None, 
                         name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                          |
| ------- | ------------ | --------------------------------------------- |
| -   | y            | 输入的 Tensor，支持的数据类型为 bool, int8, int16, in32, in64, float32, float64。 |
| -   | out            | 指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Tensor。默认值为 None，此时将创建新的 Tensor 来保存输出结果。 |
