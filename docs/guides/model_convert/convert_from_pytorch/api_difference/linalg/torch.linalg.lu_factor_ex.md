## [ torch 参数更多 ]torch.linalg.lu_factor_ex

### [torch.linalg.lu_factor_ex](https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor_ex.html?highlight=lu_factor_ex#torch.linalg.lu_factor_ex)

```python
torch.linalg.lu_factor_ex(A, *, pivot=True, check_errors=False, out=None)
```

### [paddle.linalg.lu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_cn.html)

```python
paddle.linalg.lu(x, pivot=True, get_infos=True, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                                                  |
| ------------ | ------------ | ----------------------------------------------------------------------------------------------------- |
| A            | x            | 表示需要进行 LU 分解的输入 Tensor ，仅参数名不一致。                                                  |
| pivot        | pivot        | 表示 LU 分解时是否进行旋转。                                                                          |
| -            | get_infos    | 表示是否返回分解状态信息 ，PyTorch 返回 infos 信息，Paddle 需要设置为 True。                          |
| check_errors | -            | 检查 infos 的内容，如果为非 0 抛出错误， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
| out          | -            | 表示输出的 Tensor 元组 ，Paddle 无此参数，需要转写。                                                  |
| 返回值       | 返回值       | 表示返回的 Tensor 元组 ，PyTorch 返回 info 的 shape 为[]，Paddle 返回 info 的 shape 为[1]，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.lu_factor_ex(A, out=(LU, pivots, info))

# Paddle 写法
y = paddle.linalg.lu(A, get_infos=True)
y[2] = paddle.to_tensor(y[2].item(), dtype='int32')
paddle.assign(y[0], out[0]), paddle.assign(y[1], out[1]), paddle.assign(y[2], out[2])
```

#### 返回值

```python
# Pytorch 写法
y = torch.linalg.lu_factor_ex(A)

# Paddle 写法
y = paddle.linalg.lu(A, get_infos=True)
y[2] = paddle.to_tensor(y[2].item(), dtype='int32')
```
