## [ 仅 paddle 参数更多 ]torch.nn.ZeroPad2d
### [torch.nn.ZeroPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html?highlight=zeropad#torch.nn.ZeroPad2d)

```python
torch.nn.ZeroPad2d(padding)
```

### [paddle.nn.ZeroPad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ZeroPad2D_cn.html)

```python
paddle.nn.ZeroPad2D(padding,
                    data_format='NCHW',
                    name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding   | padding | 表示填充大小。                   |
| -   | data_format | 指定输入的 format， PyTorch 无此参数， Paddle 保持默认即可。                  |
