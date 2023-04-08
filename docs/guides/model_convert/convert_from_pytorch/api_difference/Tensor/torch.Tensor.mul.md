## [ 仅 paddle 参数更多 ] torch.Tensor.mul

### [torch.Tensor.mul](https://pytorch.org/docs/1.13/generated/torch.Tensor.mul.html)

```python
torch.Tensor.mul(value)
```

### [paddle.Tensor.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#multiply-y-axis-1-name-none)

```python
paddle.Tensor.multiply(y,
                axis=-1,
                name=None)
```

两者功能一致，输入一个 y ,将 Tensor 与 y 的对应元素相乘。其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------- |
| value         | y            | 相乘的矩阵                                               |
| -             | axis         | 计算的维度， PyTorch 无此参数， Paddle 设为 -1 即可        |
