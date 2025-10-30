## [ 仅 API 调用方式不一致 ]torch.min

### [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html)

```python
torch.min(input, *, out=None)
```

### [paddle.compat.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/min_cn.html#paddle/compat/min_cn#cn-api-paddle-compat-min)

```python
paddle.compat.min(input, *args, out=None, **kwargs)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.min(x)

# Paddle 写法
result = paddle.compat.min(x)

```
