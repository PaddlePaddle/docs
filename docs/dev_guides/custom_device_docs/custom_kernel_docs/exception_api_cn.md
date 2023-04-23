# Exception API


## PADDLE_ENFORCE

使用方式：

```c++
 PADDLE_ENFORCE_{TYPE}(cond_a, // 条件 A
                       cond_b, // 条件 B, 根据 TYPE 可选
                       phi::errors::{ERR_TYPE}("{ERR_MSG}"));
```

根据`TYPE`的不同，分为：

| Exception 宏 | 判断条件 | 报错信息 |
|---|---|---|
| PADDLE_ENFORCE_EQ | cond_a == cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_NE | cond_a != cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_GT | cond_a > cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_GE | cond_a >= cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_LT | cond_a < cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_LE | cond_a <= cond_b | 触发 ERR_TYPE 异常和报 ERR_MSG |
| PADDLE_ENFORCE_NOT_NULL | cond_a != nullptr | 触发 ERR_TYPE 异常和报 ERR_MSG |

`ERR_TYPE`支持：

| 类型 | 含义 |
|---|---|
| InvalidArgument | 非法参数 |
| NotFound | 未找到 |
| OutOfRange | 越界 |
| AlreadyExists | 已存在 |
| ResourceExhausted | 资源超限 |
| PreconditionNotMet | 前置条件未满足 |
| PermissionDenied | 权限限制 |
| ExecutionTimeout | 超时 |
| Unimplemented | 未实现 |
| Unavailable | 不可用 |
| Fatal | Fatal 错误 |
| External | 外部错误 |

`ERR_MSG`为 C 语言风格字符串，支持变长参数。

示例：

```c++
// 如果 num_col_dims >= 2 && num_col_dims <= src.size()不为 true 则报 InvalidArgument 异常
// 和打印相关提示信息
PADDLE_ENFORCE_EQ(
      (num_col_dims >= 2 && num_col_dims <= src.size()),
      true,
      phi::errors::InvalidArgument("The num_col_dims should be inside [2, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size(),
                                   num_col_dims));
```

## 相关内容

- `PADDLE_ENFORCE`：请参照[enforce.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/enforce.h)
- `errors`：请参照[errors.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/errors.h)
