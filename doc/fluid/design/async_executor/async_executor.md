## Motivation of this work

There are many deep learning applications that use sparse features as inputs, such as sentiment analysis[1], word2vec[2], click through rate estimation[3]. Two characteristics exist in these applications: 1) large amount of training data exist in real world, especially in industrial environment. 2) input sparse features may not overlap in large ratio between data replicas if we use data-parallelism training method given large amount of training data. The two characteristics lead to an interesting problem of how to speed up data-parallel deep learning model with large amount of sparse features. A famous algorithm is Hogwild[4] proposed before the rise of deep learning. The authors of Hogwild state that stochasitic gradient descent algorithms can be implemented in lock-free mode that allows processors access to shared memory of model parameters and is able to over-write each-other's work. The authors show that when the associated optimization problem is sparse, Hogwild! can achieve a nearly optimal rate of convergence. In this work, we will implement an executor that can support Hogwild like update for deep learning training. Serveral experiments on natural language processing models will be conducted to show efficiency and convergence properties of the proposed executor.

## User Interface Design
``` python
import paddle as paddle
import paddle.fluid as fluid

startup_program = fluid.default_startup_program()
main_program = fluid.default_main_program()

paddle.async_executor(startup_program=startup_program, 
                      main_program=main_program)
```

## Data Feeding Approach
TBA

## Inside Structure of Async Executor
TBA

## How to print variable information during execution
TBA

## How to save models
TBA

## references
1. [Sentiment Analysis](https://arxiv.org/pdf/1801.07883.pdf)
2. [Word2Vec](https://arxiv.org/abs/1301.3781)
3. [Click Through Rate Estimation](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
4. [Hogwild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)