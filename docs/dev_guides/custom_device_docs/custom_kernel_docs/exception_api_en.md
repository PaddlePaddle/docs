# Exception API


## PADDLE_ENFORCE

How to use：

```c++
 PADDLE_ENFORCE_{TYPE}(cond_a, // Condition A
                       cond_b, // Condition B, optional based on the TYPE
                       phi::errors::{ERR_TYPE}("{ERR_MSG}"));
```

There are different conditions according to `TYPE`：

| Exception Macro | Basis | Error |
|---|---|---|
| PADDLE_ENFORCE_EQ | cond_a == cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_NE | cond_a != cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_GT | cond_a > cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_GE | cond_a >= cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_LT | cond_a < cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_LE | cond_a <= cond_b | Raise ERR_TYPE exception and report ERR_MSG |
| PADDLE_ENFORCE_NOT_NULL | cond_a != nullptr | Raise ERR_TYPE exception and report ERR_MSG |

`ERR_TYPE` supports：

| Type |
|---|
| InvalidArgument |
| NotFound |
| OutOfRange |
| AlreadyExists |
| ResourceExhausted |
| PreconditionNotMet |
| PermissionDenied |
| ExecutionTimeout |
| Unimplemented |
| Unavailable |
| Fatal |
| External |

`ERR_MSG` is a C-style string C, supporting variable-length arguments.

Example：

```c++
// If num_col_dims >= 2 && num_col_dims <= src.size() is not true, report the InvalidArgument exception.
// Print relevant tips
PADDLE_ENFORCE_EQ(
      (num_col_dims >= 2 && num_col_dims <= src.size()),
      true,
      phi::errors::InvalidArgument("The num_col_dims should be inside [2, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size(),
                                   num_col_dims));
```

## Relevant Information

- `PADDLE_ENFORCE`：please refer to [enforce.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/enforce.h)
- `errors`：please refer to [errors.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/errors.h)
