# Recognising-Textual-Entailment-using-Decomposable-Attention
This repo contains my implementation code to the paper of **A Decomposable Attention Model for Natural Language Inference**  for Recognising Textual Entailment Challenges.
## Problem statement:
Textual Entailment Recognition was proposed recently as a generic task that captures major semantic inference needs across many natural language processing applications, such as Question Answering (QA), Information Retrieval (IR), Information Extraction (IE), and (multi) document summarization. This task requires to recognise, given two text fragments, whether the meaning of one text is entailed (can be inferred) from the other text or contradict with the other text or no relation between them.
[read more @Recognising Textual Entailment Challenge](https://www.k4all.org/project/recognising-textual-entailment-challenge/).
- read the paper: [click here](https://arxiv.org/pdf/1606.01933.pdf).
- get the data: [click here](https://nlp.stanford.edu/projects/snli/).
- model Architecture: ![model arch](https://github.com/fatma-mohamed-98/Recognising-Textual-Entailment-using-Decomposable-Attention/blob/main/model_Arch.png).
- Results:
  - Test Accuracy on SNLI dataset: %83.80.

   - Train/Validation loss: \
   ![loss](https://github.com/fatma-mohamed-98/Recognising-Textual-Entailment-using-Decomposable-Attention/blob/main/train_valid_loss.png).
  - Train/Validation Acc: \
  ![acc](https://github.com/fatma-mohamed-98/Recognising-Textual-Entailment-using-Decomposable-Attention/blob/main/train_valid_acc.png).
