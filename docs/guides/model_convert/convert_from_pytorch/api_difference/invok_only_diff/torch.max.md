## [ 仅 API 调用方式不一致 ]torch.max

### [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html)

```python
torch.max(*args, **kwargs)
```

### [paddle.compat.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/max_cn.html#paddle/compat/max_cn#cn-api-paddle-compat-max)

```python
paddle.compat.max(*args, **kwargs)
```

torch.max 有多种签名与用法，均只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.max(x)

# Paddle 写法
result = paddle.compat.max(x)

```
