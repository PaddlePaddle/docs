## [ 组合替代实现 ]torch.cuda.is_available

### [torch.cuda.is_available](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch-cuda-is-available)

```python
torch.cuda.is_available()
```

### [paddle.device.cuda.device_count](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/device_count_cn.html#device-count)

```python
paddle.device.cuda.device_count()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# torch 写法
torch.cuda.is_available()

# paddle 写法
paddle.device.cuda.device_count() >= 1
```
