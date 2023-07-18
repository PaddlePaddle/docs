## [torch 参数更多]torch.nn.functional.interpolate

### [torch.nn.functional.interpolate](https://pytorch.org/docs/1.13/generated/torch.nn.functional.interpolate.html#torch.nn.functional.interpolate)

```python
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
```

### [paddle.nn.functional.interpolate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html)

```python
paddle.nn.functional.interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False, align_mode=0, data_format='NCHW', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                | PaddlePaddle  | 备注                                                                                                   |
| ---------------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| input                  | x             | 输入的 Tensor，仅参数名不一致。                                                                        |
| size                   | size          | 输出 Tensor 形状。                                                                                     |
| scale_factor           | scale_factor  | 输入的高度或宽度的乘数因子。                                                                           |
| mode                   | mode          | 插值方法。                                                                                             |
| align_corners          | align_corners | 一个可选的 bool 型参数，如果为 True，则将输入和输出张量的 4 个角落像素的中心对齐，并保留角点像素的值。 |
| recompute_scale_factor | -             | 是否重新计算 scale_factor，Paddle 无此参数，暂无转写方式。                                             |
| antialias              | -             | 是否使用 anti-aliasing，Paddle 无此参数，暂无转写方式。                                                |
| -                      | align_mode    | 双线性插值的可选项，PyTorch 无此参数，Paddle 保持默认即可。                                            |
| -                      | data_format   | 指定输入的数据格式，PyTorch 无此参数，Paddle 保持默认即可。                                            |
