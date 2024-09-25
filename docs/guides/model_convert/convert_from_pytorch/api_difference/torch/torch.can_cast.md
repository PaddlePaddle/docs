## [功能缺失]torch.can_cast

### [torch.can_cast](https://pytorch.org/docs/stable/generated/torch.can_cast.html#torch-can-cast)

```python
torch.can_cast(from_, to)
```

判断类型的转换在 PyTorch 的[casting 规则](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc)中是否被允许。

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.can_cast(x, y)

# Paddle 写法
def can_cast(from_, to):
    can_cast_dict = {
        paddle.bfloat16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False
        },
        paddle.float16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False,
        },
        paddle.float32: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False,
        },
        paddle.float64: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False,
        },
        paddle.complex64: {
            paddle.bfloat16: False,
            paddle.float16: False,
            paddle.float32: False,
            paddle.float64: False,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False,
        },
        paddle.complex128: {
            paddle.bfloat16: False,
            paddle.float16: False,
            paddle.float32: False,
            paddle.float64: False,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: False,
            paddle.int8: False,
            paddle.int16: False,
            paddle.int32: False,
            paddle.int64: False,
            paddle.bool: False,
        },
        paddle.uint8: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: False,
        },
        paddle.int8: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: False,
        },
        paddle.int16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: False,
        },
        paddle.int32: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: False,
        },
        paddle.int64: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: False,
        },
        paddle.bool: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        }
    }
    return can_cast_dict[from_][to]

can_cast(x, y)
```
