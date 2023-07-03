## [ 功能缺失 ] torch.Tensor.mul_
### [torch.Tensor.mul_](https://pytorch.org/docs/1.13/generated/torch.Tensor.mul_.html?highlight=mul_)

```python
torch.Tensor.mul_(other)
```

### [paddle.Tensor.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#multiply-y-axis-1-name-none)

```python
paddle.Tensor.multiply(y,
                axis=-1,
                name=None)
```

其中，Paddle 相比 PyTorch 支持更多其他参数，且不支持原地操作，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------- |
| other         | y            | 相乘的矩阵，仅参数名不一致。                               |
| -             | axis         | 计算的维度， PyTorch 无此参数， Paddle 保持默认即可。       |
