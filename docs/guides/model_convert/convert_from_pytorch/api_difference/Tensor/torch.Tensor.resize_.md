## [ torch 参数更多 ]torch.Tensor.resize_

### [torch.Tensor.resize_](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html?highlight=resize_#torch.Tensor.resize_)

```python
torch.Tensor.resize_(*sizes, memory_format=torch.contiguous_format)
```
### [paddle.Tensor.reshape_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id14)

```python
paddle.Tensor.reshape_(shape, name=None)
```


Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | ------- |
| *sizes      | shape        | 目标尺寸 |
| memory_format | -        | 目标内存格式，Paddle 暂无实现方法 |

### 转写示例

```python
# Pytorch 写法
y = a.resize_(1, 3, 2)

# Paddle 写法
y = a.reshape_([1, 3, 2])
```
