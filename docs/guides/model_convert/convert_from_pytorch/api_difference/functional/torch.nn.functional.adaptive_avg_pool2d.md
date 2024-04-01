## [ 仅参数名不一致 ]torch.nn.functional.adaptive_avg_pool2d

### [torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool2d.html?highlight=adaptive_avg_pool2d#torch.nn.functional.adaptive_avg_pool2d)

```python
torch.nn.functional.adaptive_avg_pool2d(input, output_size)
```

### [paddle.nn.functional.adaptive_avg_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/adaptive_avg_pool2d_cn.html)

```python
paddle.nn.functional.adaptive_avg_pool2d(x,
                                         output_size,
                                         data_format='NCHW',
                                         name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| output_size           | output_size           | 表示输出 Tensor 的大小。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
