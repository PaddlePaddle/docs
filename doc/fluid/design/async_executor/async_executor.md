## Motivation of this work

There are many deep learning applications that use sparse features as inputs, such as sentiment analysis[1], word2vec[2], click through rate estimation[3]. Two characteristics exist in these applications: 1) large amount of training data exist in real world, especially in industrial environment. 2) input sparse features may not overlap in large ratio between data replicas if we use data-parallelism training method given large amount of training data. The two characteristics lead to an interesting problem of how to speed up data-parallel deep learning model with large amount of sparse features. A famous algorithm is Hogwild[4] proposed before the rise of deep learning. The authors of Hogwild state that stochasitic gradient descent algorithms can be implemented in lock-free mode that allows processors access to shared memory of model parameters and is able to over-write each-other's work. The authors show that when the associated optimization problem is sparse, Hogwild! can achieve a nearly optimal rate of convergence. In this work, we will implement an executor that can support Hogwild like update for deep learning training. Serveral experiments on natural language processing models will be conducted to show efficiency and convergence properties of the proposed executor.

## User Interface Design
``` python
def train_loop():
    filelist = ["file%d.txt" % i for i in range(10)]
    dataset = MultiSlotDataset() # a datafeeddesc of Dataset
    dataset.set_batch_size(128) # datafeed should be assigned a batch size
    # input text data
    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    avg_cost, acc, prediction = bow_net(data, label)
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)
    opt_ops, weight_and_grad = sgd_optimizer.minimize(avg_cost)

    for w in weight_and_grad[0]:
        reduce(lambda x * y, 1, w.shape)

    varnames = [var.name for var in weight_and_grad[0]]
    dataset.set_field_name([data.name, label.name])
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    infer_prog = get_infer_prog([data.name, label.name], [acc, predict])

    place = fluid.CPUPlace()
    executor = fluid.AsyncExecutor()
    executor.run_startup_program(startup_program)
    epochs = 10
    for i in range(epochs):
        acc_val = executor.run(
            program=main_program, # make sure this can be changed during iteration
            reader=dataset, # make sure this can be changed during iteration
            filelist=filelist, # this can be changed during iteration
            thread=thread_num, # make sure this can be changed during iteration
            fetch=[acc]) # fetch can be done with python, but the scope should be exposed
        print("accuracy %f" % acc_val)
        executor.save_model(infer_prog, "epoch%d.model" % i)

    # todo: 
    # inference to be added, should loadup a program and a global scope
```
## Difference between async_executor and other executors
async_executor is mainly designed for cpu training scenarios where data throughputs are high and the computation part of training is not intensive compared with GPU trained models such as resnet-50. Since data throughputs ability is very important in async_executor, we have to design very fast data IO modules to handle very large scale data reading. Another different key aspect is that memory is not a problem in cpu training scenarios given 128G or 256G RAW in modern server. 

executor and parallel_executor are designed for geneneral training cases in particular for gpu training. Executor is a single thread implementation for model training and it is mostly used for startup_program running currently. Another application scenario of executor is reinforcement learning where input data and main_program may change through training. Parallel_executor is mainly designed for synchronous training on high performance devices such as gpu. Operators are executed concurrently following topological orders on different graphs and model parameter gradients are synchrounized iteratively.

## Data Feeding Approach
![Data Feeding Approach](https://github.com/guru4elephant/FluidDoc/blob/develop/doc/fluid/design/async_executor/async_executor_reader_design.png)
to be discussed. 

## Main Interface of Async Executor
We have RunFromFiles interface which is an execution interface for users to call. Every time a user calls RunFromFiles, a main_program should be provided and it is running in the global scope previously defined. A list of file names and corresponding Dataset should be provided. Inside the RunFromFiles interface, readers will be created through Dataset configurations. Files will be fed into created readers. 
``` c++
void AsyncExecutor::RunFromFiles(
    const ProgramDesc& main_program,
    const DataFeedDesc& data_feed_desc,
    const std::vector<std::string> & files,
    const int thread_num) {
  // todo: remove fluid related interface
  root_scope_->DropKids();
  std::vector<std::thread> threads;
  threads.resize(thread_num);

  /*
    readerDesc: protobuf description for reader initlization
    argument: class_name, batch_size, use_slot, queue_size, buffer_size, padding_index
    
    reader: 
    1) each thread has a reader, reader will read input data and 
    put it into input queue
    2) each reader has a Next() iterface, that can fetch an instance
    from the input queue
   */
  std::vector<std::shared_ptr<DataFeed> > readers;
  readers.resize(thread_num);
  for (int i = 0; i < readers.size(); ++i) {
    readers[i] = DataFeedFactory::CreateDataFeed(data_feed_desc.name());
  }

  // todo(dongdaxiang): add the following code for worker generalization
  /*
  std::vector<std::shared_ptr<ExecutorStrategy> > workers;
  workers.resize(thread_num);
  std::string str_name = strategy_.name;
  for (auto& worker : workers) {
    worker.reset(
        ExecutorStrategyFactory::CreateExecutorStrategy(str_name));
  }
  */

  std::vector<std::shared_ptr<ExecutorThreadWorker> > workers;
  workers.resize(thread_num);
  for (auto& worker : workers) {
    worker.reset(new ExecutorThreadWorker);
  }

  // prepare thread resource here
  for (int thidx = 0; thidx < thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program,
                  readers[thidx].get(), root_scope_, thidx);
  }
  // start executing ops in multiple threads
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    threads.push_back(std::thread(&ExecutorThreadWorker::TrainFiles,
                                  workers[thidx].get()));
  }

  for (auto& th : threads) {
    th.join();
  }
  // fetch variables in scope 0, and return, to be added
}

```
Inside the function ```CreateThreads```, 
``` c++
void AsyncExecutor::CreateThreads(const ExecutorThreadWorker* worker,
                                  const ProgramDesc& main_program,
                                  const DataFeed& reader,
                                  const Scope& root_scope,
                                  const int thread_index) {
  worker->SetThreadid(thread_index);
  worker->CreateThreadResource(main_program, place_);
  worker->SetDataFeed(reader);
  worker->BindingDataFeedMemory(reader);
  worker->SetRootScope(root_scope);
}

```
Inside the function ```Trainfiles```, 
``` c++
void ExecutorThreadWorker::TrainFiles() {
  // todo: configurable
  SetDevice();
  thread_reader_->Start(); // start reading thread within reader
  while (int cur_batch = thread_reader_->Next()) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }
    thread_scope_->DropKids();
  }
}

```

## How to print variable information during execution
Inside async_executor, no information is printed. Variable can be fetched through an execution of async_executor. The fetched variables can be printed through python. 

## How to save models
Models can be saved between execution of async_executor through io.save method. 

## POC experiments
### Text Classification
* network configuration
* data preparation
* performance and accuracy

### Text Matching
* network configuration
* data preparation
* performance and accuracy

## references
1. [Sentiment Analysis](https://arxiv.org/pdf/1801.07883.pdf)
2. [Word2Vec](https://arxiv.org/abs/1301.3781)
3. [Click Through Rate Estimation](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
4. [Hogwild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
