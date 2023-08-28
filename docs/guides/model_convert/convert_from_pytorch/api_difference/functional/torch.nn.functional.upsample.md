## [ 仅 paddle 参数更多 ] torch.nn.functional.upsample

### [torch.nn.functional.upsample](https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample.html?highlight=upsample#torch.nn.functional.upsample)

```python
torch.nn.functional.upsample(input,
                        size=None,
                        scale_factor=None,
                        mode='nearest',
                        align_corners=None)
```

### [paddle.nn.functional.upsample](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/upsample_cn.html#upsample)

```python
paddle.nn.functional.upsample(x,
                        size=None,
                        scale_factor=None,
                        mode='nearest',
                        align_corners=False,
                        align_mode=0,
                        data_format='NCHW',
                        name=None)
```

两者功能一致，其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入 Tensor，仅参数名不一致。      |
| size           | size           | 指定输出 Tensor 的大小 。               |
| scale_factor           | scale_factor           |  指定缩放比例 。              |
| mode           | mode           | 插值方法。支持"bilinear"或"trilinear"或"nearest"或"bicubic"或"linear"或"area" 。               |
| align_corners           | align_corners           |  双线性插值的可选项 。               |
| -           | align_mode           | 表示对输入 Tensor 运算的轴, PyTorch 无此参数， Paddle 保持默认即可。           |
| -          | data_format           | 表示输入的数据格式, PyTorch 无此参数， Paddle 保持默认即可。               |
