from transformers import AutoTokenizer, set_seed

from model import RewardModel, PolicyModel
from config import config
from PPOTrainer import PPOTrainer
from dataset import IMDBDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def train():
    """
    Helper method for training the policy model
    """
    
    set_seed(42)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize models
    policy_model = PolicyModel(config, trainable=True)
    reward_model = RewardModel(config)

    # Load data
    IMDB = IMDBDataset(tokenizer)
    dataset = IMDB.load_data(split='train')

    trainer = PPOTrainer(policy_model, reward_model, dataset, tokenizer, config)

    total_steps = (config.num_rollouts // config.batch_size) * config.ppo_epochs* config.epochs
    tbar = tqdm(initial=0, total=total_steps)
    all_scores = []

    for epoch in range(config.epochs): 
        score = trainer.train_step()
        all_scores.append(score)
        tbar.set_description(f"| score: {score:.3f} |")
        print(f"Epoch {epoch + 1} completed")

    plt.plot(all_scores)
    plt.savefig(config.save_dir)

    model_name = "gpt2_policy_model_epoch_" + str(config.ppo_epochs)
    policy_model.save_model(config.save_dir, model_name)
    print("Training complete!")