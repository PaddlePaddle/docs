# CINN 后端 Ir 数据结构概览

## 一、数据结构概况

Ir 着重学习两部分：数据和运算，例如 Datatype，Operation

重点 Ir，层级自顶向下包括：Module、Block、ScheduleBlock、ScheduleBlockRealize、For、Load、Store、Var、\_Tensor\_

对于本文未梳理的 Ir，可以查阅 ir_printer.h 头文件，查看相应 Ir 的打印信息

## 二、继承关系梳理

本文采用 ADT（Algebraic Data Type，代数数据类型）对 CINN 后端 Ir 的继承关系进行梳理。

ADT 和继承的对应关系可以理解为：

- ADT：IrNode = ExprNode | FunctionBase | IntrinsicOp 表示 IrNode 是由 ExprNode | FunctionBase | IntrinsicOp 组成的 sum type

- 继承关系：IrNode 是基类，ExprNode | FunctionBase | IntrinsicOp 是 IrNode 的三个派生类

后续两个代码框分别总结了 IrNode 和 IrNodeRef 与其对应子类的派生关系：

```
IrNode = ExprNode
       | FunctionBase
       | IntrinsicOp

ExprNode = _Buffer_
         | _BufferRange_
         | Cast
         | Let
         | Call
         | _Var_
         | Reduce
         | Select
         | Load
         | Store
         | Alloc
         | Free
         | IfThenElse
         | For
         | PolyFor
         | Ramp
         | Broadcast
         | Product
         | Sum
         | Block
         | ScheduleBlock
         | ScheduleBlockRealize
         | _Module_
         | PrimitiveNode
         | _Tensor_
         | IntImm
         | UIntImm
         | FloatImm
         | StringImm
         | UnaryOpNode
         | BinaryOpNode

FunctionBase = _Operation_

IntrinsicOp = BufferGetDataHandle
            | BufferGetDataConstHandle
            | PodValueToX
            | BufferCreate
            | GetAddr
            | ArgsConstruct
            | BuiltinIntrin

UnaryOpNode = Minus | Not

BinaryOpNode = Add
             | Sub
             | Mul
             | Div
             | Mod
             | Min
             | Max
             | EQ
             | NE
             | LT
             | LE
             | GT
             | GE
             | And
             | Or
             | FracOp

_Operation_ = PlaceholderOp
            | CallOp
            | PrecedingViewOp
            | BufferShareOp
            | ComputeOp
```

```
IrNodeRef = Module
          | Buffer
          | BufferRange
          | LoweredFunc
          | Var
          | Tensor
          | Expr
          | FunctionRef

FunctionRef = Operation
```

## 三、子类 Ir 详解

### 3.1 Module

Module 可以理解为一个编译的基本单元

```
struct _Module_ : public ExprNode<_Module_> {
  std::string name;
  Target target;
  // 【所有 Kernel 的输入输出，待确认】
  std::vector<Expr> buffers;
  // 【一个 function 对应一个 Loweredfunc，即一个 Kernel】
  std::vector<Expr> functions;
  // 【当前字段使用不多】
  std::vector<Expr> submodules;
};
```

### 3.2 Block、ScheduleBlock、ScheduleBlockRealizeIr

**疑问：** Block、ScheduleBlock 和 ScheduleBlockRealize 的关系如何理解？

**回答：** 最外层是 Block，Block 里面的每个 stmt 都是 ScheduleBlockRealize，也可以是深层的 For 循环。ScheduleBlockRealize 里面有 ScheduleBlock 字段。schedule 过程中修改的是 ScheduleBlockRealize 内的 iter_values。ScheduleBlockRealize 和 ScheduleBlock 一般是在 for 循环的最内层

```
  serial for (i_fused, 0, 16)
  {
    ScheduleBlock(var_36)
    {
      // 【i0_14 是用于 Load 和 Store 的下标】
      i0_14 = axis.bind(i_fused)
      var_36[i0_14] = (var_5[i0_14] * factor_1[i0_14])
    }
  }
```
```
struct Block : public ExprNode<Block> {
  std::vector<Expr> stmts;
};
```
```
// ScheduleBlockRealize is used to execute ScheduleBlock with the binding
// iter_values
struct ScheduleBlockRealize : public ExprNode<ScheduleBlockRealize> {
  // values of the iter_vars
  // 【i_fused】【iter_values 是循环变量 i_fused 的仿射函数】
  std::vector<Expr> iter_values;
  Expr schedule_block;
};
```
```
// ScheduleBlock is the unit of schedule IR which represents tensor's
// computation
struct ScheduleBlock : public ExprNode<ScheduleBlock> {
  // 【i0_14】
  std::vector<Var> iter_vars;
  // BufferRange(s) which is read in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  // 【var_5，factor_1】【读取的 buffer】
  std::vector<Expr> read_buffers;
  // BufferRange(s) which is written in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  // 【var_36】【写入的 buffer】
  std::vector<Expr> write_buffers;
  // Additional attributes about this schedulable block,
  // which take some auxiliary hints for future transformations.
  std::map<std::string, attr_t> attrs;
  std::string name;
  // 【Store 节点，可以理解为等号，表达 var_36[i0_14] = (var_5[i0_14] * factor_1[i0_14]) 】
  Expr body;
};
```
### 3.3 For
```
struct For : public ExprNode<For>, public ForBase {
  //! The loop variable.
  Var loop_var;
  //【循环下界】! The minimum value of the iteration.
  Expr min;
  //【循环上界】! The extent of the iteration.
  Expr extent;
  //【body 一定是个 Block{}，Block 内可以有多条语句】
  Expr body;

  DeviceAPI device_api;

  LLVMForLoopMeta metadata;
};

struct ForBase {
 private:
  // 【串行并行向量化】
  ForType for_type_{ForType::Serial};
  VectorizeInfo vectorize_info_;
  // 【为 CUDA 服务，bind 到 block 和 thread 的 idx】
  BindInfo bind_info_;
};
```
### 3.4 Load 和 Store

注意：Load 里面实际上有两个成员变量：Expr tensor（来自 LoadStoreAddrMnger） 和 std::vector<Expr> indices

```
/**
 * Load the value from a buffer (as an array).
 */
struct Load : public ExprNode<Load>, public LoadStoreAddrMnger {
  std::vector<Expr> indices;
};

struct LoadStoreAddrMnger {
  Expr tensor;  // Should be a tensor or a scalar.
  //! Tell whether the address is a tensor.
  bool is_addr_tensor() const;
  //! Tell whether the address is a scalar.
  bool is_addr_scalar() const;
};
```
```
/**
 * Store a `value` to the buffer at a given `index`.
 */
struct Store : public ExprNode<Store>, public LoadStoreAddrMnger {
  Expr value;
  // 【下标】【indices.size() == len(tensor.shape)】
  std::vector<Expr> indices;
};
```

### 3.5 Tensor、Buffer 和 Var

**疑问：** Tensor、Buffer 和 Var 的区别？

**回答：** Tensor 像是 Buffer 的一个 View，例如多个 Tensor 可以指向同一个 Buffer。_Var_指迭代量，可以理解为标量的一个值

```
class _Buffer_ : public ExprNode<_Buffer_> {
 public:
  //! The shape of the buffer.
  std::vector<Expr> shape;
  //! The strides of each dimension.
  // This can be empty, indicating that the array is contiguous.
  std::vector<Expr> strides;
  //! The name of the buffer.
  std::string name;
  //! The storage scope of the buffer, empty if global.
  std::string scope;
  //! The offset in terms of number of dtype elements (including lanes).
  Expr elem_offset;
  //! Factor of elem_offset field.
  // elem_offset is guaranteed to be multiple of offset_factor.
  int offset_factor{0};
  //! The place the buffer locates.
  Target target{UnkTarget()};
  //! Aignment requirement of data pointer in bytes.
  mutable int data_alignment{0};
  //! The memory type of the buffer.
  MemoryType memory_type{MemoryType::Heap};

  //! The data type of the elements.
  //! This is different from `type`, a buffer's type should always be
  //! `cinn_buffer_t*`.
  Type dtype;
};
```

注意：_Tensor_里面包含成员变量 Buffer

```
/**
 * _Tensor_ holds the content of a Tensor.
 *
 * NOTE(All) Some rules:
 *
 * 1. a _Tensor_ is a node in SSA, so every tensor's name should be unique,
 * 2. never try to change a tensor's name, that will cause chaos.
 */
class _Tensor_ : public ExprNode<_Tensor_> {
 public:
  //! Shape of this tensor(buffer).
  std::vector<Expr> shape;
  //! The domain of each axis(without reduce_axis)
  // TODO(Superjomn) support ISL domain.
  std::vector<Expr> domain;

  std::vector<Var> reduce_axis;
  //! The operation that generates Tensor.
  FunctionRef operation;
  //! Name of this tensor.
  std::string name;
  //! The bound buffer, for each tensor if it is not inline.
  Buffer buffer;
  //! Normal axis.
  mutable std::vector<Var> axis_;

  std::vector<Expr> new_indices{};
};
```

注意：_Var_只作为迭代量，可以理解为标量的一个值，类似于 int a = 1，其中_Var_.name = a，包含 is_reduce_axis 信息

```
/**
 * Variable used as iterator value or bound definition.
 */
struct _Var_ : public ExprNode<_Var_> {
  std::string name;
  // 【当 Var 作为下标去 Load 和 Store 时，当前下标是否为 Reduce 的】
  bool is_reduce_axis{false};
  //! Lower bound and upper bound of a axis.
  // @{
  // 【迭代量的最小值】
  Expr lower_bound;
  // 【迭代量的最大值】
  Expr upper_bound;
  // @}

  // ! Extra tag of this variable/axis.
  std::string tag;
};
```

### 3.6 其他 Ir

本节并未囊括后端所有 Ir，在学习的过程中随时更新，本文档共同建设。

```
Cast() : ExprNode(1) {}
```

```
struct Let : public ExprNode<Let> {
  Expr symbol;
  Expr body;
};
```

注意：Call 有成员变量 FunctionRef

```
struct Call : public ExprNode<Call> {
  explicit Call(Type t) : ExprNode<Call>(t) {}

  //! The name of the function/intrinsic.
  std::string name;
  //! The arguments.
  std::vector<Expr> read_args;
  std::vector<Expr> write_args;
  //! the attribute of this CallNode.
  std::map<std::string, attr_t> attrs;
  //! Type of calls.
  CallType call_type;
  //! The function to be called.
  FunctionRef func;
  //! The output value index if func's value is a tuple.
  int value_index{-1};
};
```

Reduce 中 init ：加法的初始值是 0，而乘法的初始值是 1

```
struct Reduce : public ExprNode<Reduce> {
  enum ReduceType {
    kSum = 0,
    kSub,
    kMul,
    kDiv,
    kMax,
    kMin,
    kAll,
    kAny,
  };

  //! The initial value.
  Expr init;

  // ! The body.
  Expr body;

  utils::SmallVector<Var, 4> reduce_axis;

  //! The type of the reduce operation.
  ReduceType reduce_type;
};
```

**疑问：** Alloc 中 condition 和 body 成员变量的语义？

```
/**
 * Allocate a buffer with the given type and size. The buffer lives for at most
 * the duration of the body statement, within which it is freed.
 */
struct Alloc : public ExprNode<Alloc> {
  //! The destination of the allocation, this might be a buffer or a variable.
  Expr destination;
  //! Dimensions of this buffer (as a multi-dimensional array).
  std::vector<Expr> extents;
  // NOTE the condition might be undefined, that means always true.
  Expr condition;
  // NOTE the body might be undefined, that means no specific logic other than
  // default.
  Expr body;
};
```

**疑问：** 离开一个 ScheduleBlock 后，需要显示调用 Free，释放 Alloc 出来的结果吗？

```
/**
 * Free the resources associated with the given buffer.
 */
struct Free : public ExprNode<Free> {
  Expr destination;
};
```

注意：IfThenElse 和 Select 的语义好像有重复？

```
struct Select : public ExprNode<Select> {
  Expr condition;
  Expr true_value;
  Expr false_value;
};
```
```
struct IfThenElse : public ExprNode<IfThenElse> {
  Expr condition;
  Expr true_case;
  Expr false_case;
};
```

注意：PolyFor 已经废弃

```
//! Polyhedral forloop, which condition is more complex than the normal `For`.
struct PolyFor : public ExprNode<PolyFor>, public ForBase {
  //! The iterator variable.
  Var iterator;
  // Initial value of the iterator.
  Expr init;
  //! The condition to continue the loop.
  Expr condition;
  //! Increase the iterator.
  Expr inc;
  //! The forloop body.
  Expr body;

  DeviceAPI device_api;
};
```

疑问：Ramp 对应的语义？base、stride、lanes 的语义？

```
//! A linear ramp node.
struct Ramp : public ExprNode<Ramp> {
  Expr base, stride;
  int lanes;
};
```

```
//! A vector with `lanes` elements and all of them are `value`.
struct Broadcast : public ExprNode<Broadcast> {
  Expr value;
  int lanes;
};
```

```
struct Product : public ExprNode<Product> {
  using ExprNode<Product>::operand;
};

struct Sum : public ExprNode<Sum> {
  using ExprNode<Sum>::operand;
};
```

**疑问：** PrimitiveNode 的语义？

```
/**
 * \brief PrimitiveNode holds the concept of Primitive in CINN.
 * A Primitive is a basic Call to some Expr function, it is introduced to create
 * several level of coarsed-grained IR nodes for better IR optimization and
 * hardware adaption.
 */
struct PrimitiveNode : public ExprNode<PrimitiveNode> {
  std::string name;
  //! the inputs of the PrimitiveNode, the vector<vector<Expr>> can hold
  //! variadic arguments.
  std::vector<std::vector<Expr>> arguments;
  //! the attribute of this PrimitiveNode.
  std::map<std::string, attr_t> attrs;
};
```
