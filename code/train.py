from model import RewardModel, PolicyModel
from config import config
from PPOTrainer import PPOTrainer
from dataset import IMDBDataset

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
    trainer.train()

    print("Training complete!")