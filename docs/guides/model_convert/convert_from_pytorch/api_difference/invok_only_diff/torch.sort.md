## [ 仅 API 调用方式不一致 ]torch.sort

### [torch.sort](https://pytorch.org/docs/stable/generated/torch.sort.html)

```python
torch.sort(input, dim=-1, descending=False, *, stable=False, out=None)
```

### [paddle.compat.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/sort_cn.html#paddle/compat/sort_cn#cn-api-paddle-compat-sort)

```python
paddle.compat.sort(input, dim=-1, descending=False, stable=False, out=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.sort(a)

# Paddle 写法
result = paddle.compat.sort(a)

```
