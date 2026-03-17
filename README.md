# Comment Study with AI Models
This project built a transformer encoder model to classify whether a comment is toxic with the following dataset:
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

# Feature and Label
Features for the model is the column "comment_text" in the data. There is a colum "toxicity" in the data and a binary label is defined as 1 if toxicity>0.5 otherwise 0.

# Model Architecture
Since the comment text as whole determines the toxicity, one word in the text could pay attention to all other words. So this is a scenario where self-attention is applicable. Hence the transformer encoder model is suitable for this task. The following hyperparameters are chosen:

dimention for token embedding: 32

number of attention heads: 4

number of attention layers: 3

These hyperparameters are configurable in the code. The vocabulary size is 50,257 (it is the same as the size from the 'gpt2' tokenizer from the tiktoken library). And the maximum sequence window size is determined by the maximum number of tokens among all comments in the training data.

# Code Structure
There are two sets of code. One if for training the model with one process and the other one is for distributed training with data parallel over multiple processes (e.g., multiple GPUs). There structures are similar. Below are the structure for the one-process training code:

models.py: code for the model architecture (transformer encoder)

utilsData.py: dataset and dataloader (how the comment texts are tokenized and converted to token IDs)

utilsTrain.py: functions for running the training job

utilsEval.py: functions for evaluating the model (at a checkpoint)

main.py: entry point for starting training job for the model

# How to Run the Code
For the one-process job:
Go to the folder "transformer" and run the following command:

CUDA_VISIBLE_DEVICES="0" python main.py
(change to CUDA_VISIBLE_DEVICES="-1" if there is no GPU)

For the multi-process job:
Go to the foler "transformer_distributed" and run the following command:

torchrun --nproc_per_node=4 main_distributed.py

"torchrun" is the utility for running distributed jobs with pytorch (similar to mpirun in HPC).

# How to Choose the Final Model
The model will be trained for a certain number of epochs. At each epoch, the model will be evaluated on a validation dataset with the metric of area under the precision-recall curve. After all the epochs, the model corresponding the epoch at which the AUC is highest will be considered as the best model. And then the best model will be tested on the testing dataset. (Note that the best model cannot be determined by evaluating each epoch on the testing dataset, as doing so is essentially "sneakily" make the model fit well just on the testing dataset and not necessarily generalize well.)

# Train-Validate-Test Data Split
To split the data into training set, validation set and testing set, two methods can be applied:

1. Randomly select 60% of the data for training, 20% for validation, and 20% for testing. (Order of the data does not matter.)

2. Sort the data by increasing time. And then use the first 60% for training, the next 20% for validation and the last 20% for testing. (This is to mimic how we usually built a deep learning application in real-world. In real-world, we train a model on the data we have up to a certain date, and then apply the model to run inference on real-time data in the future. So naturally there is time order in the training data and inference data.)

# How to Handle Data Imbalance Issue
When the data size ratio for class 0 and class 1 is very imbalanced, there are a few options:

1. Downsampling the major class: This might help with the training, however, if too much downsampling is applied, then the population is modified too much, and a model trained on the modified population might perform badly when it's applied to real-world data from the original true population.

2. Use generative models such as VAE, GAN, etc. to artificially generate data points for the minor class to mitigate the imbalance issue.

4. For text data, we can use LLMs to enhance data points for the minor class. For example, in this project if the toxic comments are much fewer than normal comments, then we can feed each toxic comment into an LLM such as ChatGPT or Gemini and instruct the LLM to "expand" or "shorten" or "rephrase" the orginal comment and add these expanded/shortened/rephrased comments to the original data so that the positive class (toxic comment class) have more representations in the training data, so that the imbalance issue can be mitigated. (For validatation and testing, they should still be performed on the original datasets.)