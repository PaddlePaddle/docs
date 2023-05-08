## [ torch 参数更多 ]torch.linalg.lu

### [torch.linalg.lu](https://pytorch.org/docs/1.13/generated/torch.linalg.lu.html?highlight=torch+linalg+lu#torch.linalg.lu)

```python
torch.linalg.lu(A, *, pivot=True, out=None)
```

### [paddle.linalg.lu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/lu_cn.html)

```python
paddle.linalg.lu(x, pivot=True, get_infos=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| A     | x           | 表示需要进行 LU 分解的输入 Tensor ，仅参数名不一致。                         |
| pivot       | pivot        | 表示 LU 分解时是否进行旋转。                           |
| -     | get_infos           | 表示是否返回分解状态信息。                         |
| out           | -      | 表示输出的三个 Tensor 元组 ， Paddle 无此参数，需要进行转写。         |

###  转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.linalg.solve_triangular(A, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.triangular_solve(A) , y)
```
