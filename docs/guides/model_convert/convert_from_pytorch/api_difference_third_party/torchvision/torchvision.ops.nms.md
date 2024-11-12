## [paddle 参数更多]torchvision.ops.nms

### [torchvision.ops.nms](https://pytorch.org/vision/main/generated/torchvision.ops.nms.html)

```python
torchvision.ops.nms(boxes: Tensor, scores: Tensor, iou_threshold: float)
```

### [paddle.vision.ops.nms](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/ops/nms_cn.html)

```python
paddle.vision.ops.nms(boxes, iou_threshold=0.3, scores=None, category_idxs=None, categories=None, top_k=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注    |
| --------------------------- | ------------------------------ | --------------------- |
| boxes                       | boxes                            | 待进行计算的框坐标。       |
| scores                      | scores                       | 与 boxes 参数对应的 score。       |
| iou_threshold               | iou_threshold                | 用于判断两个框是否重叠的 IoU 门限值。       |
| -                           | category_idxs                | 与 boxes 参数对应的类别编号，PyTorch 无此参数，Paddle 保持默认即可。       |
| -                           | categories                   | 类别列表，PyTorch 无此参数，Paddle 保持默认即可。       |
| -                           | top_k                        | 需要返回的分数最高的 boxes 索引数量，PyTorch 无此参数，Paddle 保持默认即可。     |
