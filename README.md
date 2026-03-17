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
