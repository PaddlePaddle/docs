## [ 参数不一致 ]torch.nn.UpsamplingNearest2d

### [torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html?highlight=upsampl#torch.nn.UpsamplingNearest2d)

```python
torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)
```

### [paddle.nn.UpsamplingNearest2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/UpsamplingNearest2D_cn.html)

```python
paddle.nn.UpsamplingNearest2D(size=None,scale_factor=None, data_format='NCHW',name=None)
```

其中 Paddle 和 PyTorch 的`size`参数支持类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | size         | 表示输出 Tensor 的 size ，PyTorch 支持一个单独的数或元组或列表，Paddle 仅支持元组和列表，不支持一个单独的数，需要转写。                                     |
| scale_factor           | scale_factor            | 表示输入 Tensor 的高度或宽度的乘数因子。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
### 转写示例
#### size：输出 Tensor 的 size
```python
# Pytorch 写法
model = nn.UpsamplingNearest2d(size=4)

# Paddle 写法
model = paddle.nn.UpsamplingNearest2D(size=(4, 4))
