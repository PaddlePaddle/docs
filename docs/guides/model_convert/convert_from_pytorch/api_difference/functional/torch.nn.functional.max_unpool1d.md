## [ 仅参数名不一致 ]torch.nn.functional.max_unpool1d

### [torch.nn.functional.max_unpool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_unpool1d.html?highlight=max_unpool1d#torch.nn.functional.max_unpool1d)

```python
torch.nn.functional.max_unpool1d(input,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 output_size=None)
```

### [paddle.nn.functional.max_unpool1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/max_unpool1d_cn.html)

```python
paddle.nn.functional.max_unpool1d(x,
                                 indices,
                                 kernel_size,
                                 stride=None,
                                 padding=0,
                                 data_format='NCL',
                                 output_size=None,
                                 name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| indices           | indices           | 表示索引下标。               |
| kernel_size           | kernel_size           | 表示滑动窗口大小。               |
| stride           | stride           | 表示步长。               |
| padding           | padding           | 表示填充大小。               |
| output_size           | output_size           | 表示目标输出尺寸。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
