## torch.nn.Upsample
### [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html?highlight=upsample#torch.nn.Upsample)

```python
torch.nn.Upsample(size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=False)
```

### [paddle.nn.Upsample](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Upsample_cn.html#upsample)

```python
paddle.nn.Upsample(size=None,
                   scale_factor=None,
                   mode='nearest',
                   align_corners=False,
                   align_mode=0,
                   data_format='NCHW',
                   name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | align_mode   | 双线性插值的可选项，PyTorch 无此参数，Paddle 保持默认即可。    |
| -             | data_format  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |
