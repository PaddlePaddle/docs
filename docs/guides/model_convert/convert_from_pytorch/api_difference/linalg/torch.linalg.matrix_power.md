## [ torch 参数更多 ]torch.linalg.matrix_power
### [torch.linalg.matrix_power](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_power.html?highlight=torch+linalg+matrix_power#torch.linalg.matrix_power)

```python
torch.linalg.matrix_power(input,
                        n,
                        *,
                        out=None)
```

### [paddle.linalg.matrix_power](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_power_cn.html)

```python
paddle.linalg.matrix_power(x,
                        n,
                        name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          |  x           | 输入的欲进行 n 次幂运算的一个或一批方阵，类型为 Tensor,仅参数名不一致。  |
| n         | n         | 输入的幂次，类型为 int。 |
|out         | -         |  表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.linalg.matrix_power(x, 3， out = y)

# Paddle 写法
paddle.assign(paddle.linalg.matrix_power(x, 3), y)
```
