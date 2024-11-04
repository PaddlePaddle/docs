## [输入参数用法不一致]torchvision.ops.roi_pool

### [torchvision.ops.roi_pool](https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html)

```python
torchvision.ops.roi_pool(input: Tensor, boxes: Union[Tensor, List[Tensor]], output_size: None, spatial_scale: float = 1.0)
```

### [paddle.vision.ops.roi_pool](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/roi_pool_cn.html)

```python
paddle.vision.ops.roi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None)
```

两者功能一致，ROIs 框坐标 `boxes` 的用法不一致：
在 PyTorch 中，boxes 参数可以通过一个形状为 (K, 5) 的 Tensor，或者一个 Tensor List 来指定每个 ROI 的批次索引和坐标。
1. 当 boxes 是形状为 (K, 5) 的 Tensor 时:
- 每一行的第一个数值表示该 ROI 属于输入 batch 中的第几个样本(批次索引)。
- 每一行的后四个数值表示该 ROI 的坐标 [x1, y1, x2, y2]。
例如 boxes[i] = [0, 4, 4, 7, 7] 表示第 i 个 ROI 属于第 0 个样本,其坐标为 (4, 4, 7, 7)
2. 当 boxes 是 Tensor List 时:
- List 的长度等于 batch size。
- List 中的每个 Tensor 形状为 (L, 4),表示该样本中所有 ROI 的坐标。
- 每个 ROI 坐标由 4 个数值组成 [x1, y1, x2, y2]。
例如 boxes[0] 是形状为 (L, 4) 的 Tensor,表示第 0 个样本中的 L 个 ROI 坐标

而在 Paddle 中，这个功能被拆分为两个参数：
1. boxes: 仅包含 ROI 的坐标信息，形状为 (num_rois, 4)，每一行是一个 ROI 的 [x1, y1, x2, y2] 坐标。
2. boxes_num: 一个形状为 (batch_size) 的 Tensor，用于指定每个样本中包含的 ROI 数量。

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| input                                 | x                  | 输入特征图，仅参数名不一致。|
| boxes                                 | boxes, boxes_num   | 待执行池化的 ROIs 的框坐标，Paddle 用 boxes 和 boxes_num 等价的实现 PyTorch boxes 参数的功能，需要转写。|
| output_size                           | output_size        | 池化后输出的尺寸。|
| spatial_scale                         | spatial_scale      | 空间比例因子。|

### 转写示例
#### boxes：待执行池化的 ROIs 的框坐标
boxes 是一个形状为 (K, 5) 的二维 Tensor 时
```python
# PyTorch 写法
boxes = torch.tensor([[0, 4, 4, 7, 7], [1, 5, 5, 10, 10]], dtype=torch.float32)
output = torchvision.ops.roi_pool(input, boxes=boxes, output_size=7, spatial_scale=1.0)

# Paddle 写法
boxes = paddle.to_tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype='float32')
boxes_num = paddle.to_tensor([1, 1], dtype='int32')
output = paddle.vision.ops.roi_pool(x, boxes=boxes, boxes_num=boxes_num, output_size=7, spatial_scale=1.0)
```

boxes 是 Tensor List 时
```python
# PyTorch 写法
boxes = [torch.tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype=torch.float32)]
torchvision.ops.roi_pool(input, boxes=boxes, output_size=7, spatial_scale=1.0)

# Paddle 写法
boxes = paddle.to_tensor([[4, 4, 7, 7], [5, 5, 10, 10]], dtype='float32')
boxes_num = paddle.to_tensor([1, 1], dtype='int32')
output = paddle.vision.ops.roi_pool(x, boxes=boxes, boxes_num=boxes_num, output_size=7, spatial_scale=1.0)
```
