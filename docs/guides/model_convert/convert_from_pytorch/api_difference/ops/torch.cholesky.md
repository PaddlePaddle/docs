## [仅 torch 参数更多]torch.cholesky

### [torch.cholesky](https://pytorch.org/docs/stable/generated/torch.cholesky.html?highlight=cholesky#torch.cholesky)

```python
torch.cholesky(input,upper=False,*,out=None)
```

### [paddle.linalg.cholesky](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cholesky_cn.html)

```python
paddle.linalg.cholesky(x,upper=False,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|x|输入变量为多维 Tensor，它的维度应该为 [*, M, N]，其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵|
|upper|upper|指示是否返回上三角矩阵或下三角矩阵|
|out||输出矩阵，Paddle无此变量，需要进行转写|

### 转写示例

```python
# Pytorch 写法
>>> a = torch.randn(3, 3)
>>> a = a @ a.mT + 1e-3
>>> l = torch.cholesky(a, upper=False, out=y)
>>> l
tensor([[ 1.5528,  0.0000,  0.0000],
        [-0.4821,  1.0592,  0.0000],
        [ 0.9371,  0.5487,  0.7023]])

# Paddle 写法
 
import paddle

a = paddle.rand([3, 3], dtype="float32")
a_t = paddle.transpose(a, [1, 0])
x = paddle.matmul(a, a_t) + 1e-03

out = paddle.linalg.cholesky(x, upper=False)
print(out)

```
