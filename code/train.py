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
    
    # Initialize models
    policy_model = PolicyModel(config, trainable=True)
    reference_model = PolicyModel(config, trainable=False)
    reward_model = RewardModel(config)

    # Load data
    IMDB = IMDBDataset(policy_model.tokenizer)
    dataset = IMDB.load_data(split='train')

    trainer = PPOTrainer(policy_model, reference_model, reward_model, dataset)

    tbar = tqdm(initial=0, total=config.epochs)
    all_scores = []

    for epoch in range(config.epochs): 
        score = trainer.train_step()
        all_scores.append(score)
        tbar.set_description(f"| Epoch: {epoch+1}, Score: {score:.3f} |")
        tbar.update()

    plt.plot(all_scores)
    plt.savefig(config.save_dir+"Rewards.png")

    model_name = 'GPT2-RLHF-PPO-EPOCH-'+ str(config.epochs)
    PPOTrainer.save_trained_model(config.save_dir, model_name=model_name)
    
    print("Training complete!")