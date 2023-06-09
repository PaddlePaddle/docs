## [ 组合替代实现 ]torch.Tensor.resize_

### [torch.Tensor.resize_](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html?highlight=resize_#torch.Tensor.resize_)

```python
torch.Tensor.resize_(*sizes, memory_format=torch.contiguous_format)
```
Paddle 无此 API，需要组合实现, 对于 memory_format 不为 torch.contiguous_format 的情况，暂无转写方式。

### 转写示例
#### memory_format 为 torch.contiguous_format 时
```python
# Pytorch 写法
x.resize_(2, 3, 3)

# Paddle 写法
num = 1
for ele in [2, 3, 3]:
    num *= ele
paddle.assign(paddle.flatten(x)[:num].reshape([2, 3, 3]), x)
```
