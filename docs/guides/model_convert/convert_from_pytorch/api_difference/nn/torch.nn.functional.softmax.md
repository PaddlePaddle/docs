## [torch 参数更多]torch.nn.functional.softmax

### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)

```python
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softmax_cn.html#softmax)

```python
paddle.nn.functional.softmax(x, axis=- 1, dtype=None, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                  |
|:-------:|:------------:| :---------------------------------------------------: |
| input   |   x           |  表示输入张量，仅参数名不一致。           |
| dim     | axis         |  表示对输入 Tensor 进行运算的轴，仅参数名不一致。            |
| dtype   |   dtype      |  表示返回张量所需的数据类型。  |
| - | name | 一般无需设置，默认值为 None， PyTorch 无此参数。 |




### 转写示例
```python
# PyTorch 写法
import torch
import torch.nn.functional as f

x = torch.Tensor([[[2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]]])
out = f.softmax(x, dim = 1, dtype='float32')


# Paddle 写法
import paddle
import paddle.nn.functional as F

x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]]],dtype='float32')
out = F.softmax(x)
```
