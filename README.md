# RLHF with Proximal Policy Optimization (PPO)

This project implements **Reinforcement Learning with Human Feedback (RLHF)** using **Proximal Policy Optimization (PPO)** to fine-tune a large language model (a pre-trained GPT-2 from Huggingface `lvwerra/gpt2-imdb`) based on human preferences. The reward model (`siebert/sentiment-roberta-large-english`) is used to guide the policy updates, improving the quality of text generation based on human-aligned objectives.

## Project Overview

This project uses **PPO** to optimize the text generation of GPT-2, guided by a reward model that reflects human feedback. The main steps include:

- **Policy Model (GPT-2)**: The language model that generates text based on prompts.
- **Reward Model (`siebert/sentiment-roberta-large-english`)**: Evaluates the generated responses, assigning a score based on sentiment or human preferences.
- **PPO Algorithm**: Updates the GPT-2 model by maximizing the reward signal while ensuring stable updates with clipping.

## How It Works

1. **Language Model (Policy)**:
   - GPT-2 generates text token by token, and each token prediction is considered an action.
   - The model learns to generate meaningful and coherent sequences by optimizing over multiple training epochs.

2. **Reward Model**:
   - The reward model (using `siebert/sentiment-roberta-large-english`) evaluates the text generated by the language model. It assigns rewards based on how well the output aligns with positive sentiments or other desired outcomes.
   
3. **PPO Training**:
   - The Proximal Policy Optimization (PPO) algorithm is used to fine-tune the language model by maximizing the rewards while ensuring stable training.
   - The policy model is updated in a way that prevents drastic changes, using advantage estimation and importance sampling.

4. **Training Data**:
   - The IMDB dataset is used for both the reward model training and the input prompts for the GPT-2 model.

## Files and Structure

- `train.py`: Contains the training loop that fine-tunes the GPT-2 model using PPO. After training, the model is saved to the `saved_models` folder.
- `generate.py`: Generates text sequences using the trained model based on user input.
- `PPOTrainer.py`: Implements the PPO algorithm, including advantage estimation and policy updates.
- `model.py`: Defines the policy model (`PolicyModel`) and the reward model (`RewardModel`).
- `dataset.py`: Loads and processes the IMDB dataset.
- `config.py`: Contains the configuration parameters used for training (e.g., learning rate, batch size).
- `main.py`: The entry point for running training or generation tasks using command-line arguments.

## Setup and Installation

To run the training:

```bash
python main.py --task train
```

### Requirements

- Python 3.8+
- PyTorch
- Hugging Face `transformers` library
- `datasets` library
- torchtyping

You can install the dependencies using `pip` or using the requirements.yml file:

```bash
conda env create -f requirements.yml
```