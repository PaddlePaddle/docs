## [ 组合替代实现 ]torch.logdet

### [torch.logdet](https://pytorch.org/docs/stable/generated/torch.logdet.html#torch.logdet)

```python
torch.logdet(input)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.logdet(input)

# Paddle 写法
y = paddle.log(paddle.linalg.det(input))
```
