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

# How to run the code
For the one-process job:
Go to the folder "transformer" and run the following command:

CUDA_VISIBLE_DEVICES="0" python main.py
(change to CUDA_VISIBLE_DEVICES="-1" if there is no GPU)

For the multi-process job:
Go to the foler "transformer_distributed" and run the following command:

python --nproc_per_node=4 main_distributed.py


