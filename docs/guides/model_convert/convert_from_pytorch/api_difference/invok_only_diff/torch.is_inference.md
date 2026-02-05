## [ 仅 API 调用方式不一致 ]torch.is_inference

### [torch.is_inference](https://pytorch.org/docs/stable/generated/torch.is_inference.html)

```python
torch.is_inference(input)
```

### [paddle.Tensor.stop\_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.is_inference(x)

# Paddle 写法
result = x.stop_gradient
```
