## [ 仅 paddle 参数更多 ]torch.Tensor.softmax
### [torch.Tensor.softmax](https://pytorch.org/docs/stable/generated/torch.Tensor.softmax.html?highlight=softmax#torch.Tensor.softmax)

```python
torch.Tensor.softmax(dim)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softmax_cn.html#softmax)

```python
paddle.nn.functional.softmax(x, axis=-1, dtype=None, name=None)
```

功能一致，torch 是类成员方式，paddle 是 function 调用，具体差异如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 计算 softmax 的轴                                         |

### 转写示例
#### dim: 计算 softmax 的轴
```python
# torch 写法
x.softmax(dim=1)

# paddle 写法
paddle.nn.functional.softmax(x, axis=1)
```
