## torch.nn.AdaptiveMaxPool1d
### [torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/1.13/generated/torch.nn.AdaptiveMaxPool1d.html?highlight=adaptivemaxpool1d#torch.nn.AdaptiveMaxPool1d)

```python
torch.nn.AdaptiveMaxPool1d(output_size,
                           return_indices=False)
```

### [paddle.nn.AdaptiveMaxPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AdaptiveMaxPool1D_cn.html#adaptivemaxpool1d)

```python
paddle.nn.AdaptiveMaxPool1D(output_size,
                            return_mask=False,
                            name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| return_indices| return_mask  | 如果设置为 True，则会与输出一起返回最大值的索引，默认为 False。 |
