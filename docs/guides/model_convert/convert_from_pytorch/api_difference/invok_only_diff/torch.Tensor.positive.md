## [ 仅 API 调用方式不一致 ]torch.Tensor.positive

### [torch.Tensor.positive](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.positive.html#torch.Tensor.positive)

```python
torch.Tensor.positive()
```

### [paddle.positive](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/positive_cn.html#paddle.positive)

```python
paddle.positive(x)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.positive()

# Paddle 写法
result = paddle.positive(x)
```
