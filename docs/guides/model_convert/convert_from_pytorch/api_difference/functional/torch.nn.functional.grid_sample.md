## [ 仅参数名不一致 ] torch.nn.functional.grid_sample

### [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html?highlight=grid_sample#torch.nn.functional.grid_sample)

```python
torch.nn.functional.grid_sample(input,
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=None)
```

### [paddle.nn.functional.grid_sample](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/grid_sample_cn.html)

```python
paddle.nn.functional.grid_sample(x,
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True,
                        name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor，仅参数名不一致。               |
| grid           | grid           |  输入网格数据张量。               |
| mode           | mode           |   指定插值方式。               |
| padding_mode           | padding_mode           |   指定超出边界的填充方式。               |
| align_corners           | align_corners           |   是否将角落的点进行中心对齐。     |
