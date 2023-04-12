## [ 仅 paddle 参数更多 ] torch.Tensor.mul_
### [torch.Tensor.mul_](https://pytorch.org/docs/1.13/generated/torch.Tensor.mul_.html?highlight=mul_)

```python
torch.Tensor.mul_(value)
```

### [paddle.Tensor.scale_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#id16)

```python
paddle.Tensor.scale_(scale=1.0,
                bias=0.0,
                bias_after_scale=True,
                act=None,
                name=None)
```
Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle     | 备注                                                                      |
| ------------- | ---------------- | --------------------------------------------------------------------------|
| value         | scale            | 放缩的大小，仅参数名不一致                                                   |
| -             | bias             | 表示放缩后的偏置部分， PyTorch 无此参数， paddle 保持默认即可。               |
| -             | bias_after_scale | 表示是否在放缩后加偏置部分， PyTorch 无此参数， paddle 保持默认即可。         |
| -             | act              | PyTorch 无此参数， paddle 保持默认即可。                                   |
