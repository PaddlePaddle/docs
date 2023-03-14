## [ 仅参数名不一致 ]torch.nn.AdaptiveMaxPool3d
### [torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/1.13/generated/torch.nn.AdaptiveMaxPool3d.html?highlight=adaptivemaxpool3d#torch.nn.AdaptiveMaxPool3d)

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

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| output_size | output_size  | 表示输出 Tensor 的大小。 |
| return_indices| return_mask  | 如果设置为 True，则会与输出一起返回最大值的索引，默认为 False，仅参数名不一致。 |
