## [ 仅 API 调用方式不一致 ]torch.Tensor.is_inference

### [torch.Tensor.is\_inference](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_inference.html#torch.Tensor.is_inference)

```python
torch.Tensor.is_inference()
```

### [paddle.Tensor.stop\_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.is_inference()

# Paddle 写法
result = x.stop_gradient
```
