## [ 组合替代实现 ]torch.Tensor.logdet

### [torch.Tensor.logdet](https://pytorch.org/docs/stable/generated/torch.Tensor.logdet.html#torch.Tensor.logdet)

```python
torch.Tensor.logdet()
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = input.logdet()

# Paddle 写法
y = paddle.log(paddle.linalg.det(input))
```
