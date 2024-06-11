## [ torch 参数更多 ]torch.nn.parameter.Parameter
### [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter)

```python
torch.nn.parameter.Parameter(data=None,
                             requires_grad=True)
```

### [paddle.create_parameter](https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/fluid/layers/tensor.py#L77)

```python
paddle.create_parameter(shape,
                        dtype,
                        name=None,
                        attr=None,
                        is_bias=False,
                        default_initializer=None)
```

两者功能一致但参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| data          | -            | 参数 Tensor，Paddle 无此参数。  |
| requires_grad | -            | ，Paddle 无此参数。  |
| -             | shape        | 指定输出 Tensor 的形状，PyTorch 无此参数。  |
| -             | dtype        | 初始化数据类型，PyTorch 无此参数。  |
| -             | attr         | 指定参数的属性对象，PyTorch 无此参数。  |
| -             | is_bias      | 当 default_initializer 为空，该值会对选择哪个默认初始化程序产生影响。如果 is_bias 为真，则使用 initializer.Constant(0.0)，否则使用 Xavier()，PyTorch 无此参数。  |
| -             | default_initializer | 参数的初始化程序，PyTorch 无此参数。  |

### 功能差异

#### 使用方式
***PyTorch***：通过设置`data`将 Tensor 赋给 Parameter。
***PaddlePaddle***：有 2 种方式创建 Parameter。方式一：通过设置`attr`将 ParamAttr 赋给 Parameter；方式二：通过设置`shape`（大小）、`dtype`（类型）、`default_initializer`（初始化方式）设置 Parameter。

#### 梯度设置
***PyTorch***：通过设置`requires_grad`确定是否进行梯度反传。
***PaddlePaddle***：PaddlePaddle 无此功能。


### 代码示例
``` python
# PyTorch 示例：
import torch
x = torch.zeros(2, 3)
param = torch.nn.parameter.Parameter(x, requires_grad=False)

# 输出
# Parameter containing:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

``` python
# PaddlePaddle 示例：
import paddle
x = paddle.zeros([2, 3], dtype="float32")
param = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
param.stop_gradient = True

# 输出
# Parameter containing:
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0., 0., 0.],
#         [0., 0., 0.]])
```
