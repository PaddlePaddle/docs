## [ 组合替代实现 ]torch.Tensor.addcdiv_

### [torch.Tensor.addcdiv_](https://pytorch.org/docs/stable/generated/torch.Tensor.addcdiv_.html#torch.Tensor.addcdiv_)

```python
torch.Tensor.addcdiv_(tensor1, tensor2, *, value=1)
```

用于实现矩阵 `tensor1` 与矩阵 `tensor2` 相除，再加上输入 `input` ，公式为：

$ out =  input + value *  (tensor1 / tensor2) $

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例

```python
# PyTorch 写法
input.addcdiv_(tensor1, tensor2, value=value)

# Paddle 写法
paddle.assign(input + value * tensor1 / tensor2, input)
```
