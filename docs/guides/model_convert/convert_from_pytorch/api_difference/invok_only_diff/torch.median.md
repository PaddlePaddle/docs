## [ 仅 API 调用方式不一致 ]torch.median

### [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html)

```python
torch.median(input)
```

### [paddle.compat.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/median_cn.html#paddle/compat/median_cn#cn-api-paddle-compat-median)

```python
paddle.compat.median(input, dim=None, keepdim=False, *, out=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.median(input)

# Paddle 写法
result = paddle.compat.median(input)

```
