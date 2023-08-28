## [ 仅 paddle 参数更多 ]torch.nn.functional.prelu

### [torch.nn.functional.prelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.prelu.html?highlight=prelu#torch.nn.functional.prelu)

```python
torch.nn.functional.prelu(input,
                          weight)
```

### [paddle.nn.functional.prelu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/prelu_cn.html)

```python
paddle.nn.functional.prelu(x,
                           weight,
                           data_format='NCHW',
                           name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| weight           |  weight           | 表示激活公式中的可训练权重 。               |
| -           |  data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
