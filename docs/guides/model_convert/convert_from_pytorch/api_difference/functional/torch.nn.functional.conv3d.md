## torch.nn.functional.conv3d

### [torch.nn.functional.conv3d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html?highlight=conv3d#torch.nn.functional.conv3d)

```python
torch.nn.functional.conv3d(input,
                           weight,
                           bias=None,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=1)
```

### [paddle.nn.functional.conv3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/conv3d_cn.html)

```python
paddle.nn.functional.conv3d(x,
                            weight,
                            bias=None,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            data_format='NCDHW',
                            name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
