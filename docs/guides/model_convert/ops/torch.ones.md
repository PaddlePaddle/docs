## torch.ones
### [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones)

```python
torch.ones(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_cn.html#ones)

```python
paddle.ones(shape,
             dtype=None,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |


### 功能差异

#### 使用方式
***PyTorch***：生成 Tensor 的形状大小以可变参数、list 或 tuple 的方式传入。
***PaddlePaddle***：生成 Tensor 的形状大小以 list 或 tuple 的方式传入。

### 代码示例
``` python
# PyTorch 示例：
torch.ones(2, 3)
# 输出
# tensor([[ 1.,  1.,  1.],
#         [ 1.,  1.,  1.]])
```

``` python
# PaddlePaddle 示例：
paddle.ones([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[1., 1., 1.],
#         [1., 1., 1.]])
```
