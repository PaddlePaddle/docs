## [ paddle 参数更多 ]torch.nn.Upsample
### [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=upsample#torch.nn.Upsample)

```python
torch.nn.Upsample(size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=False)
```

### [paddle.nn.Upsample](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Upsample_cn.html#upsample)

```python
paddle.nn.Upsample(size=None,
                   scale_factor=None,
                   mode='nearest',
                   align_corners=False,
                   align_mode=0,
                   data_format='NCHW',
                   name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size            | size   | 表示输出 Tensor 的大小。    |
| scale_factor             | scale_factor   | 输入的高度或宽度的乘数因子。    |
| mode             | mode   | 表示插值方法。    |
| align_corners             | align_corners   | 表示是否将输入和输出张量的 4 个角落像素的中心对齐，并保留角点像素的值。    |
| -             | align_mode   | 双线性插值的可选项，PyTorch 无此参数，Paddle 保持默认即可。    |
| -             | data_format  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |
