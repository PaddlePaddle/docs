## torch.nn.AdaptiveMaxPool3d
### [torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool3d.html?highlight=nn+adaptivemaxpool3d#torch.nn.AdaptiveMaxPool3d)

```python
torch.nn.AdaptiveMaxPool3d(output_size,
                            return_indices=False)
```

### [paddle.nn.AdaptiveMaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AdaptiveMaxPool3D_cn.html#adaptivemaxpool3d)

```python
paddle.nn.AdaptiveMaxPool3D(output_size,
                            return_mask=False,
                            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| return_indices| return_mask  | 如果设置为 True，则会与输出一起返回最大值的索引，默认为 False。 |
