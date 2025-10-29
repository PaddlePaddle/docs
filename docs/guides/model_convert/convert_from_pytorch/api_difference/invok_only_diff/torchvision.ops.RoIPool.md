## [ 仅 API 调用方式不一致 ]torchvision.ops.RoIPool

### [torchvision.ops.RoIPool](https://pytorch.org/vision/stable/generated/torchvision.ops.RoIPool.html#torchvision.ops.RoIPool)

```python
torchvision.ops.RoIPool(output_size, spatial_scale)
```

### [paddle.vision.ops.RoIPool](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/ops/RoIPool_cn.html#paddle/vision/ops/RoIPool_cn#cn-api-paddle-vision-ops-RoIPool)

```python
paddle.vision.ops.RoIPool(output_size, spatial_scale)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
roi_pool = torchvision.ops.RoIPool(output_size=(7, 7), spatial_scale=1.0)

# Paddle 写法
roi_pool = paddle.vision.ops.RoIPool(output_size=(7, 7), spatial_scale=1.0)
```
