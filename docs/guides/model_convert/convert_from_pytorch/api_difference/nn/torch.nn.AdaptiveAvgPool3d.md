## [ paddle 参数更多 ] torch.nn.AdaptiveAvgPool3d

### [torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html)

```python
torch.nn.AdaptiveAvgPool3d(output_size)
```

### [paddle.nn.AdaptiveAvgPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AdaptiveAvgPool3D_cn.html#adaptiveavgpool3d)

```python
paddle.nn.AdaptiveAvgPool3D(output_size, data_format='NCDHW', name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| output_size   | output_size  | 表示输出 Tensor 的 size 。                              |
| -             | data_format  | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。 |
