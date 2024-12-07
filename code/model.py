from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import os

class PolicyModel(nn.Module):
    """
    Policy Model with value head for Alignment
    """
    def __init__(self, config, trainable):    
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainable = trainable
        self.model = AutoModelForCausalLM.from_pretrained(config.policy_model_name, torch_dtype = torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if not self.trainable:
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        else:
            n_embd = self.model.lm_head.in_features
            num_labels = 1
            self.value_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, num_labels),
            ).to(torch.bfloat16).to(self.device)
        self.logit_head = self.model.get_output_embeddings()
    
    def forward(self, input_ids, attention_mask=None):
        
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        lm_logits = self.logit_head(last_hidden_state)
        
        if self.trainable:
            value = self.value_head(last_hidden_state).squeeze(-1)
            return lm_logits, value
        else:
            return lm_logits
        
    def generate(self, input_ids, **gen_kwargs):
        return self.model.generate(input_ids, **gen_kwargs)
    
    def save_model(self, save_dir, model_name):
        parent_dir = os.path.join(os.getcwd(), os.pardir)
        model_dir = os.path.join(save_dir, model_name)
        save_dir = os.path.join(parent_dir, model_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f'Model and Tokenizer saved to {save_dir}')
    
        
class RewardModel(nn.Module):

    """
    Custom reward head on top of AutoModelForCausalLM model
    """
    
    # def __init__(self, config):
    #     super().__init__()
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     self.model = AutoModelForCausalLM.from_pretrained(config.reward_model_name, torch_dtype = torch.bfloat16).to(self.device)
    #     n_embd = self.model.config.hidden_size
    #     self.reward_head = nn.Sequential(
    #             nn.LayerNorm(n_embd),
    #             nn.GELU(),
    #             nn.Linear(n_embd, 1),
    #         ).to(torch.bfloat16).to(self.device)

    # def forward(self, input_ids, attention_mask=None):
    #     outputs = self.model(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         output_hidden_states=True)
    #     last_hidden_state = outputs.hidden_states[-1]
    #     last_hidden_state_mean = torch.mean(last_hidden_state, 1)
    #     last_hidden_state_mean = last_hidden_state_mean.to(torch.bfloat16)
    #     logits = self.reward_head(last_hidden_state_mean)
    #     return logits

    """
    AutoModelForSequenceClassification using built-in scaler head on top of base causal model
    """

    def __init__(self, config):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name, torch_dtype = torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)

    def forward(self, texts):
        
        input_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(**input_ids).logits[:, 1:]
        
        temperature = 0.3
        sentiments = torch.sigmoid(predictions*temperature).squeeze()

        return sentiments

    
