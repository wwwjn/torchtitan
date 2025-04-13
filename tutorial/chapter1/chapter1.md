# Chapter1: Adding Data Parallel (DDP, FSDP)

Data parallelism is the first step to scale up training. We will talk about basic DDP and FSDP concepts in this chapter, and scale-up Llama3.1 model with and FSDP in this chapter. We will also using experiment results to show how data parallelism can improve the training speed.


## 1.1 Distibuted Data Parallel (DDP)
Sending different batch of data to different GPUs to train the model.

## 1.2 Fully Sharded Data Parallel (FSDP)
Reduced Peak Memory Usage, in trade-off for communication overhead between GPUs.



## 1.3 Experiments
