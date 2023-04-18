## [torch 参数更多]torch.nn.functional.softmax

### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)

```python
torch.nn.functional.softmax
```

### [paddle.nn.Softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax_cn.html)

```python
paddle.nn.Softmax(axis=-1,name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                  |
|:-------:|:------------:| :---------------------------------------------------: |
| input   |   -           |  Paddle 无此参数，需要转写。                                  |
| dim     | axis         |  指定对输入 Tensor 进行运算的轴，仅参数名不一致。              |
| dtype   |   -           |  Paddle 无此参数。    |


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

x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]]], dtype='float32')
m = paddle.nn.Softmax()
out = m(x)
```

