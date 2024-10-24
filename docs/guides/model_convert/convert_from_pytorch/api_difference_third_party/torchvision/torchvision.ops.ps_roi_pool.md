## [输入参数用法不一致]torchvision.ops.ps_roi_pool

### [torchvision.ops.ps_roi_pool](https://pytorch.org/vision/main/generated/torchvision.ops.ps_roi_pool.html)

```python
torchvision.ops.ps_roi_pool(input: Tensor, boxes: Tensor, output_size: int, spatial_scale: float = 1.0)
```

### [paddle.vision.ops.psroi_pool](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/psroi_pool_cn.html)

```python
paddle.vision.ops.psroi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None)
```

两者功能一致，ROIs 框坐标 `boxes` 的用法不一致，具体如下：

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| input                                 | x                  | 输入特征图，仅参数名不一致。|
| boxes                                 | boxes, boxes_num   | 待执行池化的 ROIs 的框坐标，Paddle 的 boxes 只接受一个形状为 (num_rois, 4) 的 Tensor，用 boxes_num 指定每个样本中的 ROI 数量；PyTorch 的 boxes 可以是形状为 (K, 5) 的 Tensor，其中第一列是批次索引，其余四列是 [x1, y1, x2, y2]；也可以是一个 Tensor List，每个 Tensor 形状为 (L, 4)，表示每个样本中的所有 ROI 坐标，需要转写。|
| output_size                           | output_size        | 池化后输出的尺寸。|
| spatial_scale                         | spatial_scale      | 空间比例因子。|

### 转写示例
#### boxes：待执行池化的 ROIs 的框坐标
boxes 是 Tensor[K, 5]，其中第一列是批次索引，其余四列是 [x1, y1, x2, y2]。
```python
# PyTorch 写法
boxes = torch.tensor([[0, 4, 4, 7, 7], [1, 5, 5, 10, 10]], dtype=torch.float32)
output = torchvision.ops.ps_roi_pool(input, boxes=boxes, output_size=7, spatial_scale=1.0)

# Paddle 写法
boxes = paddle.to_tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype='float32')
boxes_num = paddle.to_tensor([1, 1], dtype='int32')
output = paddle.vision.ops.psroi_pool(x, boxes=boxes, boxes_num=boxes_num, output_size=7, spatial_scale=1.0)
```

boxes 是 Tensor List，每个 Tensor 形状为 (L, 4)，表示每个样本中的所有 ROI 坐标。
```python
# PyTorch 写法
boxes = [torch.tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype=torch.float32)]
output = torchvision.ops.ps_roi_pool(input, boxes=boxes, output_size=7, spatial_scale=1.0)

# Paddle 写法
boxes = paddle.to_tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype='float32')
boxes_num = paddle.to_tensor([1, 1], dtype='int32')
output = paddle.vision.ops.psroi_pool(x, boxes=boxes, boxes_num=boxes_num, output_size=7, spatial_scale=1.0)
```
