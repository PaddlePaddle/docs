## [仅 torch 参数更多]torch.linalg.cholesky

### [torch.linalg.cholesky](https://pytorch.org/docs/stable/generated/torch.linalg.cholesky.html?highlight=linalg+cholesky#torch.linalg.cholesky)

```python
torch.linalg.cholesky(A,*,upper=False,out=None)
```

### [paddle.linalg.cholesky](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cholesky_cn.html)

```python
paddle.linalg.cholesky(x,upper=False,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|A|x|输入的多维矩阵，它的维度应该为 [*, M, N]，其中*为零或更大的批次尺寸，并且最里面的两个维度上的矩阵都应为对称的正定矩阵|
|upper|upper|指示是否返回上三角矩阵或下三角矩阵|
|out||输出矩阵|

### 转写示例

```python
# Pytorch 写法
>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> A_t = torch.transpose(A,1,0)
>>> x = torch.matmul(A,A_t) + 1e-03
>>> torch.linalg.cholesky(x, upper=False, out=y)


# Paddle 写法
 
import paddle

a = paddle.rand([3, 3], dtype=<span class="s2">"float32")
a_t = paddle.transpose(a, [1, 0])
x = paddle.matmul(a, a_t) + 1e-03

out = paddle.linalg.cholesky(x, upper=False)
print(out)

```
