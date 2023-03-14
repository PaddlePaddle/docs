## torch.Tensor.int
### [torch.Tensor.int](https://pytorch.org/docs/stable/generated/torch.Tensor.int.html?highlight=int#torch.Tensor.int)

```python
torch.Tensor.int(memory_format)
```

### [paddle.Tensor.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#astype-dtype)

```python
paddle.Tensor.astype(dtype)
```

两者功能一致，参数名和参数不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| memory_format | -            | tensor 存储形式，转写无需考虑                                |
|-              | dtype        | 转换后的 tensor 类型                                       |

### 转写示例

```python
# torch 写法
x.int()

# paddle 写法
x.astype('int32')
```
