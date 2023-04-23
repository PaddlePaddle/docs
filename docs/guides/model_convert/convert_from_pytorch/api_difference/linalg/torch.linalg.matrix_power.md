## [ 参数不一致 ]torch.linalg.matrix_power
### [torch.linalg.matrix_power](https://pytorch.org/docs/1.13/generated/torch.linalg.matrix_power.html?highlight=torch+linalg+matrix_power#torch.linalg.matrix_power)

```python
torch.linalg.matrix_power(A,
                        n,
                        *,
                        out=None)
```

### [paddle.linalg.matrix_power](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/matrix_power_cn.html)

```python
paddle.linalg.matrix_power(x,
                        n,
                        name=None)
```

两者功能一致但参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A          |  x           | 输入的欲进行 n 次幂运算的一个或一批方阵，类型为 Tensor,仅参数名不一致。  |
| n         | n         | 输入的幂次，类型为 int。 |
|out         | -         |  表示输出的 Tensor ， Paddle 无此参数，需要进行转写。 |
| -   |   name  |        可选，一般无需设置，默认值为 None，PyTorch 无此参数， Paddle 保持默认即可。                            |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
torch.linalg.matrix_power(x, 3， out = y)

# Paddle 写法
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7,8,9]])
paddle.assign(paddle.linalg.matrix_power(x, 3), y)
```
