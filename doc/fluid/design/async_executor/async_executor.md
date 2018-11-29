## Motivation of this work

There are many deep learning applications that use sparse features as inputs, such as sentiment analysis[1], word2vec[2], click through rate estimation[3]. Two characteristics exist in these applications: 1) large amount of training data exist in real world, especially in industrial environment. 2) input sparse features may not overlap in large ratio between data replicas if we use data-parallelism training method given large amount of training data. The two characteristics lead to an interesting problem of how to speed up data-parallel deep learning model with large amount of sparse features. A famous algorithm is Hogwild[4] proposed before the rise of deep learning. The authors of Hogwild state that stochasitic gradient descent algorithms can be implemented in lock-free mode that allows processors access to shared memory of model parameters and is able to over-write each-other's work. The authors show that when the associated optimization problem is sparse, Hogwild! can achieve a nearly optimal rate of convergence. In this work, we will implement an executor that can support Hogwild like update for deep learning training. Serveral experiments on natural language processing models will be conducted to show efficiency and convergence properties of the proposed executor.

## User Interface Design
``` python
def train_loop():
    # Download data
    with tarfile.open(paddle.dataset.common.download(URL, "imdb", MD5)) as tarf:
        tarf.extractall(path='./')
        tarf.close()
     # Initialize dataset description
    dataset = fluid.DataFeedDesc('train_data/data.prototxt')
    dataset.set_batch_size(128)  # See API doc for how to change other fields
    print dataset.desc()  # Debug purpose: see what we get
     # define network
    # input text data
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
     avg_cost, acc, prediction = bow_net(data, label)
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)
    opt_ops, weight_and_grad = sgd_optimizer.minimize(avg_cost)
     # Run startup program
    startup_program = fluid.default_startup_program()
    place = fluid.CPUPlace()
    executor = fluid.Executor(place)
    executor.run(startup_program)
    main_program = fluid.default_main_program()
    epochs = 10
    filelist = ["train_data/part-%d" % i for i in range(12)]
    for i in range(epochs):
        thread_num = 4
        executor.run_from_files(
            main_program,  # This can be changed during iteration
            dataset,  # This can be changed during iteration
            filelist,  # This can be changed during iteration
            thread_num,  # This can be changed during iteration
            [data, acc],  # Multiple fetch targets can be specified
            debug=False)
        fluid.io.save_inference_model('imdb/epoch%d.model' % i,
                                      [data.name, label.name], [acc], executor)

```
## Difference between async_executor and other executors
async_executor is mainly designed for cpu training scenarios where data throughputs are high and the computation part of training is not intensive compared with GPU trained models such as resnet-50. Since data throughputs ability is very important in async_executor, we have to design very fast data IO modules to handle very large scale data reading. Another different key aspect is that memory is not a problem in cpu training scenarios given 128G or 256G RAM in modern clusters. 

executor and parallel_executor are designed for geneneral training cases in particular for gpu training. Executor is a single thread implementation for model training and it is mostly used for startup_program running currently. Another application scenario of executor is reinforcement learning where input data and main_program may change through training. Parallel_executor is mainly designed for synchronous training on high performance devices such as gpu. Operators are executed concurrently following topological orders on different graphs and model parameter gradients are synchrounized iteratively.

## Data Feeding Approach
![Data Feeding Approach](https://github.com/guru4elephant/FluidDoc/blob/develop/doc/fluid/design/async_executor/async_executor_reader_design.png)
Why we use multiple queues for data reader? a experimental result page needs to be added.

## Main Interface of Async Executor
We have RunFromFiles interface which is an execution interface for users to call. Every time a user calls RunFromFiles, a main_program should be provided and it is running in the global scope previously defined. A list of file names and corresponding Dataset should be provided. Inside the RunFromFiles interface, readers will be created through Dataset configurations. Files will be fed into created readers. 
``` c++
void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& data_feed_desc_str,
                                const std::vector<std::string>& filelist,
                                const int thread_num,
                                const std::vector<std::string>& fetch_var_names,
                                const bool debug) {
  std::vector<std::thread> threads;
   auto& block = main_program.Block(0);
  for (auto var_name : fetch_var_names) {
    auto var_desc = block.FindVar(var_name);
    auto shapes = var_desc->GetShape();
    PADDLE_ENFORCE(shapes[shapes.size() - 1] == 1,
                   "var %s: Fetched var has wrong shape, "
                   "only variables with the last dimension size 1 supported",
                   var_name);
  }
   DataFeedDesc data_feed_desc;
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc);
   int actual_thread_num = thread_num;
  int file_cnt = filelist.size();
  PADDLE_ENFORCE(file_cnt > 0, "File list cannot be empty");
   if (actual_thread_num > file_cnt) {
    VLOG(1) << "Thread num = " << thread_num << ", file num = " << file_cnt
            << ". Changing thread_num = " << file_cnt;
    actual_thread_num = file_cnt;
  }
  std::vector<std::shared_ptr<DataFeed>> readers;
  PrepareReaders(readers, actual_thread_num, data_feed_desc, filelist);
   std::vector<std::shared_ptr<ExecutorThreadWorker>> workers;
  workers.resize(actual_thread_num);
  for (auto& worker : workers) {
    worker.reset(new ExecutorThreadWorker);
  }
   // prepare thread resource here
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program, readers[thidx],
                  fetch_var_names, root_scope_, thidx, debug);
  }
   // start executing ops in multiple threads
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
    threads.push_back(
        std::thread(&ExecutorThreadWorker::TrainFiles, workers[thidx].get()));
  }
   for (auto& th : threads) {
    th.join();
  }
   root_scope_->DropKids();
   return;
}

```
Inside the function ```CreateThreads```, 
``` c++
void AsyncExecutor::CreateThreads(
    ExecutorThreadWorker* worker,
    const ProgramDesc& main_program,
    const std::shared_ptr<DataFeed>& reader,
    const std::vector<std::string>& fetch_var_names,
    Scope& root_scope,
    const int thread_index) {
  worker->SetThreadId(thread_index);
  worker->SetRootScope(&root_scope);
  worker->CreateThreadResource(main_program, place_);
  worker->SetDataFeed(reader);
  worker->SetFetchVarNames(fetch_var_names);
  worker->BindingDataFeedMemory();
}

```
Inside the function ```Trainfiles```, 
``` c++
void ExecutorThreadWorker::TrainFiles() {
  // todo: configurable
  SetDevice();

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num, 0);

  thread_reader_->Start();

  int cur_batch;
  int batch_cnt = 0;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }

    float avg_inspect = 0.0;
    for (int i = 0; i < fetch_var_num; ++i) {
      avg_inspect = thread_scope_->FindVar(fetch_var_names_[i])
                                 ->GetMutable<LoDTensor>()
                                 ->data<float>()[0];
      fetch_values_[i] += avg_inspect;
    }

    ++batch_cnt;
    thread_scope_->DropKids();
  }

  if (batch_cnt) {
    // when the number of files is less than the number of threads
    for (int i = 0; i < fetch_var_num; ++i) {
      fetch_values_[i] = fetch_values_[i] / batch_cnt;
    }
  }

```

## How to print variable information during execution
Inside async_executor, no information is printed. Variable can be fetched through an execution of async_executor. The fetched variables can be printed through python. Since we train several files of instances within async_executor, the fetched variables are not accurate. In this version of design, we only fetch variables of the last iteration for each thread and we average the fetched variables by batch_size * thread_num. 

## How to save models
Models can be saved between execution of async_executor through io.save method. 

## POC experiments
### Text Classification
* network configuration
* data preparation
* performance and accuracy

## references
1. [Sentiment Analysis](https://arxiv.org/pdf/1801.07883.pdf)
2. [Word2Vec](https://arxiv.org/abs/1301.3781)
3. [Click Through Rate Estimation](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
4. [Hogwild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
