## [ 仅 API 调用方式不一致 ]torch.nanmedian

### [torch.nanmedian](https://pytorch.org/docs/stable/generated/torch.nanmedian.html)

```python
torch.nanmedian(input)
```

### [paddle.compat.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/nanmedian_cn.html#paddle/compat/nanmedian_cn#cn-api-paddle-compat-nanmedian)

```python
paddle.compat.nanmedian(input, dim=None, keepdim=False, *, out=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.nanmedian(input)

# Paddle 写法
result = paddle.compat.nanmedian(input)

```
