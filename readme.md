
# Simple LLM from Scratch

This repository contains a simple implementation of a Language Model (LLM) built using PyTorch. The goal is to create a working transformer model from scratch to understand the underlying architecture and mechanics of large language models like GPT-2. This model is intended for educational purposes to help you learn how transformers work, including word embeddings, positional encoding, attention mechanisms, and feedforward layers.

## Overview

The model consists of several key components:
1. **Word Embeddings**: Converts tokens into high-dimensional vectors to represent words.
2. **Positional Encoding**: Adds information about the order of tokens in the sequence.
3. **Attention Mechanism**: A multi-head attention layer that computes the relevance between different tokens in the sequence.
4. **Feedforward Layer**: A simple feedforward network for processing the transformed token representations.
5. **Transformer Blocks**: Combines attention and feedforward layers with residual connections.
6. **Training**: The model is trained using a custom dataset and is optimized using the Adam optimizer and Cross-Entropy Loss.
7. **Generation**: The trained model can generate text by predicting the next token based on a given prompt.

## Installation

To run this project, you need to install the following Python dependencies:

```bash
pip install torch numpy
```

## Dataset

The model is trained on a custom dataset stored in the `tense.txt` file. Each sentence is tokenized using the GPT-2 tokenizer, and then converted into a binary format for efficient storage and access. 

You can modify the `tense.txt` file to include any dataset of your choice for training.

## Model Architecture

### 1. Word Embedding
The word embeddings layer converts integer token IDs into dense vectors. The model uses a simple embedding layer (`nn.Embedding`) for this purpose.

### 2. Positional Encoding
Since transformers are not inherently sequential (like RNNs), we add positional encoding to inject information about token order into the embeddings.

### 3. Attention
The attention layer computes the attention scores for each token and combines them based on the importance. It uses scaled dot-product attention and is masked to prevent attending to future tokens in the sequence.

### 4. Feedforward Network
Each transformer block includes a feedforward network with a ReLU activation function, followed by a linear layer to map the output back to the original embedding space.

### 5. Transformer Blocks
The transformer block is composed of a combination of the attention layer and the feedforward network. Layer normalization and residual connections are applied to both.

### 6. Generation
After training, the model can generate text by predicting the next token iteratively using the context provided as input.

## Training the Model

The model is trained using a simple loop, where each batch consists of a sequence of tokens and their corresponding targets. The training objective is to minimize the cross-entropy loss between the model's predicted token probabilities and the actual tokens in the target sequence.

```python
def train(epoch, dataloader):
    for i in range(epoch):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            out, loss = jarvis(context, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {i + 1}, Loss: {total_loss}")
```

## Text Generation

Once the model is trained, you can generate new text by providing a prompt. The model will generate tokens based on the context and output a sequence of tokens.

```python
op = jarvis.generate(torch.tensor([tokenizer.encode("The baby is")]), 10)
print(tokenizer.decode(op[0]))
```

## Saving and Loading the Model

The model's state dictionary can be saved after training and loaded later for inference.

```python
torch.save(jarvis.state_dict(), "trained.pth")
```

To load the saved model:

```python
jarvis.load_state_dict(torch.load("trained.pth"))
```

## Usage

### 1. Preprocess your data:
Prepare your dataset in a `.txt` file and call the `preprocess_sentences` function to convert the text into tokenized binary format.

### 2. Train the model:
Set the number of epochs and start the training loop.

### 3. Generate text:
After training, you can generate text by providing a prompt to the model's `generate` function.
