## [ 仅参数名不一致 ] torch.nn.functional.affine_grid

### [torch.nn.functional.affine_grid](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html?highlight=affine_grid#torch.nn.functional.affine_grid)

```python
torch.nn.functional.affine_grid(theta,
            size,
            align_corners=None)
```

### [paddle.nn.functional.affine_grid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/affine_grid_cn.html)

```python
paddle.nn.functional.affine_grid(theta,
            out_shape,
            align_corners=True,
            name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| theta          | theta         | 指定仿射变换矩阵        |
| size          | out_shape         | 表示指定目标输出图像大小，仅参数名不一致。     |
| align_corners          | align_corners         | 指定是否是像素中心对齐               |
