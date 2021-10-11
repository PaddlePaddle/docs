# API Description - Functor
Introduce the calculation functors defined by the Kernel Primitive API. There are currently 13 functors that can be used directly.

## Unary Functor

### ExpFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::ExpFunctor<Tx, Ty>();
```
#### Description
Exp operation is performed on the input of Tx type, and the result is converted into Ty type and returned.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
auto functor = kps::ExpFunctor<float>();
float input = 0;
float out = functor(input);

// out = exp(0)
// out = 1
```


### IdentityFunctor
#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::IdentityFunctor<Tx, Ty>();
```

#### Description
Convert the input data of Tx type to Ty type and return.

#### Template Parameters
> Tx : The type of input. </br>
> Ty : The type of return. </br>

#### Example
```
auto functor = kps::DivideFunctor<float, double>();
float input = 3.0f;
double out = functor(input);

// out = 3.0;
```

### DivideFunctor
#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::DivideFunctor<Tx, Ty>(num);
```

#### Description
Divide the input data of Tx type by num, and convert the result into Ty type to return.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
auto functor = kps::DivideFunctor<float>(10);
float input = 3.0f;
float out = functor(input);

// out = (3.0 / 10)
// out = 0.3
```

### SquareFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::SquareFunctor<Tx, Ty>();
```
#### Description
Perform Square operation on the input number of Tx type, and convert the result into Ty type to return.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
auto functor = kps::SquareFunctor<float>();
float input = 3.0f;
float out = functor(input);

// out = 3.0 * 3.0
// out = 9.0
```


## Binary Functor

### MinFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MinFunctor<T>();
```
#### Description
Returns the minimum of the two inputs. MinFunctor provides the initial() function for data initialization and returns the maximum value represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::MinFunctor<float>();
float input1 = 0;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 < input2 ? input1 : input2
// out = 0
```

### MaxFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MaxFunctor<T>();
```
#### Description
Returns the maximum value of the two inputs. MaxFunctor provides the initial() function for data initialization, and returns the minimum value represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::MaxFunctor<float>();
float input1 = 0;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 > input2 ? input1 : input2
// out = 1
```

### AddFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::AddFunctor<T>();
```
#### Description
Returns the sum of the two inputs. AddFunctor provides the initial() function for data initialization, and returns the data 0 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::AddFunctor<float>();
float input1 = 1;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 + input2
// out = 2
```

### MulFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MulFunctor<T>();
```
#### Description
Returns the product of two inputs. MulFunctor provides the initial() function for data initialization, and returns data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::MulFunctor<float>();
float input1 = 1;
float input2 = 2;
float out = functor(input1, input2);

// out = input1 * input2
// out = 2
```

### LogicalOrFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalOrFunctor<T>();
```
#### Description
Returns the result of the logical or operation of two inputs. LogicalOrFunctor provides the initial() function for data initialization, and returns false for the data represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::LogicalOrFunctor<bool>();
bool input1 = false;
bool input2 = true;
bool out = functor(input1, input2);

// out = input1 || input2
// out = true
```

### LogicalAndFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalAndFunctor<T>();
```
#### Description
Returns the result of the logical and operation of the two inputs. LogicalAndFunctor provides the initial() function for data initialization, and returns true for the data represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::LogicalAndFunctor<bool>();
bool input1 = false;
bool input2 = true;
bool out = functor(input1, input2);

// out = input1 && input2
// out = false
```

### SubFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::SubFunctor<T>();
```
#### Description
Two inputs are subtracted. SubFunctor provides the initial() function for data initialization, and returns the data 0 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::SubFunctor<float>();
float input1 = 1;
float input2 = 2;
float out = functor(input1, input2);

// out = input1 - input2
// out = 1
```

### DivFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::DivFunctor<T>();
```
#### Description
Two inputs are divided. DivFunctor provides the initial() function for data initialization, and returns the data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
auto functor = kps::DivFunctor<float>();
float input1 = 1.0;
float input2 = 2.0;
float out = functor(input1, input2);

// out = input1 / input2
// out = 0.5
```

### FloorDivFunctor

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::FloorDivFunctor<T>();
```
#### Description
Divide the two inputs and return the integer part. FloorDivFunctor provides the initial() function for data initialization, and returns data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example

```
auto functor = kps::FloorFunctor<float>();
float input1 = 1.0;
float input2 = 2.0;
float out = functor(input1, input2);

// out = input1 / input2
// out = 0
```
## Functor Definition Rules
In the current calculation function, only ElementwiseAny supports setting the functor parameter as a pointer, and the functor of other calculation functions can only be set as a normal parameter.

### Common parameters
Except ElementwiseAny API, other calculation functions only support ordinary parameter transfer. For example, to realize (a + b) * c, Functor can be defined as follows:

ExampleFunctor2:
```
template <typename T>
struct ExampleFunctor2 {
   inline HOSTDEVICE T operator()(const T &input1, const T &input2, const T &input3) const {
       return ((input1 + input2) * input3);
   }
};
```
### Example

```
// Global memory input pointer input0, input1, input2
auto functor = ExampleFunctor2<float>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
const int Arity = 3; // the pointers of inputs

int num = NX * NY * blockDim.x;
float inputs[Arity][NX * NY];
float output[NX * NY];

kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[0], input0, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[1], input1, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[2], input2, num);
kps::ElementwiseTernary<float, float, NX, NY, BlockSize, Arity, ExampleFunctor2<float>>(output, inpputs[0], inputs[1], inputs[2], functor);
// ...
```

### Pointer
When defining the Functor of ElementwiseAny, you need to ensure that the parameter of the operate() function is an array pointer. For example, to realize the function: (a + b) * c + d, you can combine ElementwiseAny and Functor to complete the calculation.

ExampleFunctor1:
```
template <typename T>
struct ExampleFunctor1 {
   inline HOSTDEVICE T operator()(const T * args) const { return ((arg[0] + arg[1]) * arg[2] + arg[3]); }
};
```
### example

```
// Global memory input pointer input0, input1, input2, input3
auto functor = ExampleFunctor1<float>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
const int Arity = 4; // the pointers of inputs

int num = NX * NY * blockDim.x;
float inputs[Arity][NX * NY];
float output[NX * NY];

kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[0], input0, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[1], input1, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[2], input2, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[3], input3, num);
kps::ElementwiseAny<float, float, NX, NY, BlockSize, Arity, ExampleFunctor1<float>>(output, inputs, functor);
```
