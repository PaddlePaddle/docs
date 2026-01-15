## [ 仅 API 调用方式不一致 ]torch.median

### [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html)

```python
torch.median(*args, **kwargs)
```

### [paddle.compat.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/median_cn.html#paddle/compat/median_cn#cn-api-paddle-compat-median)

```python
paddle.compat.median(*args, **kwargs)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.median(input)

# Paddle 写法
result = paddle.compat.median(input)

```
