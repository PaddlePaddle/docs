## [组合替代实现]torch.cuda.is_initialized

### [torch.cuda.is_initialized]()(https://pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html)

```python
torch.cuda.is_initialized()
```

判断 cuda 是否初始化，Paddle 无此 API，需要组合实现。
Paddle 可以通过检查是否支持 cuda，并且尝试创建一个张量来判断初始化是否成功。

### 转写示例

```python
# torch 写法
torch.cuda.is_initialized()

# paddle 写法
def paddle_cuda_is_initialized():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        cuda_tensor = paddle.rand([1], place=paddle.CUDAPlace(0))
        return True
    except Exception as e:
        return False
paddle_cuda_is_initialized()
```
