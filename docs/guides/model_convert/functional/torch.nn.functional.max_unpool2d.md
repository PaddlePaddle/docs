## torch.nn.functional.max_unpool2d

### [torch.nn.functional.max_unpool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_unpool2d.html?highlight=max_unpool2d#torch.nn.functional.max_unpool2d)

```python
torch.nn.functional.max_unpool2d(input,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 output_size=None)
```

### [paddle.nn.functional.max_unpool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/max_unpool2d_cn.html)

```python
paddle.nn.functional.max_unpool2d(x,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 data_format='NCHW',
                                 output_size=None,
                                 name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
