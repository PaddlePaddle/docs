## [paddle 参数更多]torchvision.ops.ps_roi_pool

### [torchvision.ops.ps_roi_pool](https://pytorch.org/vision/main/generated/torchvision.ops.ps_roi_pool.html)

```python
torchvision.ops.ps_roi_pool(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0
) -> Tensor
```

### [paddle.vision.ops.psroi_pool](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/psroi_pool_cn.html)

```python
paddle.vision.ops.psroi_pool(
    x: paddle.Tensor,
    boxes: paddle.Tensor,
    boxes_num: paddle.Tensor,
    output_size: Union[int, Tuple[int, int]],
    spatial_scale: float = 1.0,
    name=None
) -> paddle.Tensor
```


### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| input                                 | x                  | 输入特征图，形状为 (N, C, H, W)。|
| boxes                                 | boxes              | ROIs 的框坐标，Paddle 支持相同的 (x1, y1, x2, y2) 形式，但需要额外提供 `boxes_num` 参数。PyTorch 中 `boxes` 支持 Tensor 或 List[Tensor]，而 Paddle 仅支持 Tensor。|
| -                                     | boxes_num          | 每张图所包含的框的数量，Paddle 需要此参数以指定每个图像的框数量。|
| output_size                           | output_size        | 池化后输出的尺寸，参数含义一致。|
| spatial_scale                         | spatial_scale      | 空间比例因子，参数含义一致。|
| -                                     | name               | 一般无需设置，默认值为 None。|
