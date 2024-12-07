import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from PPORolloutStorage import PPORolloutStorage,PPORLElement

from config import config

class PPOTrainer:
    def __init__(self, policy_model, reference_model, reward_model, dataset):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.dataset = torch.tensor(dataset)
        self.tokenizer = policy_model.tokenizer
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.rollout_store = PPORolloutStorage(self.tokenizer)
        self.writer = SummaryWriter()

    def train(self):
        """
        Perform rollouts generation and update the policy model using the custom loss function
        """
        #step 1: generate and store rollouts

        pbar = tqdm(total=config.epoch, desc="Training Epochs")

        for epoch in range(config.epochs):

            torch.cuda.empty_cache()

            self.rollout_store.clear_history()
            rollouts, score = self.generate_experience()
            self.rollout_store.push(rollouts)

            train_dataloader = self.rollout_store.create_loader(config.batch_size, shuffle=True)

            #step 2: loss calculation and optimization

            self.policy_model.model.train()
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_dataloader):
                for _ in range(config.ppo_epochs):
                    loss, reward = self.loss_fn(batch)
                    loss.backward()
                    self.optimizer.step()

                # Log to TensorBoard
                global_step = epoch * len(train_dataloader) + batch_idx
                self.writer.add_scalar('Loss/train', loss.item(), global_step)
                self.writer.add_scalar('Reward/train', reward, global_step)
                self.writer.add_scalar('Score/train', score, global_step)

            pbar.update(1)
            self.scheduler.step()
            pbar.set_postfix({'loss': loss.item(), 'reward': reward, 'score': score})

        pbar.close()
        self.writer.close()                
        
        save_dir = config.save_dir
        model_name = config.policy_model_name + '-RLHF-PPO-EPS-' + config.epochs
        self.policy_model.save_model(save_dir, model_name)
    
    def generate_experience(self):
        """
        Generate the rollouts and store it in PPORolloutStorage
        """

        all_rollouts = []
        generate_kwargs = dict(
            config.gen_kwargs,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id
            )
        
        dataset_idx = np.arange(len(self.dataset))

            
        while len(all_rollouts) < config.num_rollouts:

            if len(dataset_idx) >= config.prompt_batch_size:
                picked_indices = np.random.choice(dataset_idx, config.prompt_batch_size, replace=False)
            else:
                # Handle case where there are fewer indices than `prompt_batch_size`
                picked_indices = np.random.choice(dataset_idx, len(dataset_idx), replace=False)
            
            input_ids = torch.tensor(self.dataset[picked_indices])
            dataset_idx = np.delete(dataset_idx, np.isin(dataset_idx, picked_indices).nonzero()[0])
            attention_mask = torch.ones_like(input_ids)
            batch = {'input_ids': input_ids, 'attention_mask': attention_mask}

            query_tensors = batch['input_ids'].to(self.policy_model.device)
            trajectories = self.policy_model.generate(
                query_tensors,
                attention_mask = batch['attention_mask'].to(self.policy_model.device),
                **generate_kwargs
            )
            response_tensors = trajectories[:, query_tensors.shape[1]:]
            attention_mask = trajectories.not_equal(self.tokenizer.pad_token_id).long()

            with torch.no_grad():
                logits, values = self.policy_model(
                    trajectories,
                    attention_mask = attention_mask,
                )
                ref_logits = self.reference_model(
                    trajectories,
                    attention_mask = attention_mask,
                )

            logprobs = self.logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:]) 
            ref_logprobs = self.logprobs_from_logits(ref_logits[:, :-1, :], trajectories[:, 1:])
            n_trajectories = trajectories.shape[0]
            values = values[:, :-1]

            start = batch['input_ids'].shape[1] - 1
            ends = start + attention_mask[:, start:].sum(1)
            truncated_values = [values[i, start : ends[i]] for i in range(n_trajectories)]
            truncated_logprobs = [logprobs[i, start : ends[i]] for i in range(n_trajectories)]

            texts = self.tokenizer.batch_decode(trajectories, skip_special_tokens=True)             
            scores = self.reward_model(texts)
            rewards = -config.kl_coef * (logprobs - ref_logprobs)
            all_rewards = [None] * n_trajectories

            del texts, ref_logprobs, logits, ref_logits, trajectories
            
            for i in range(n_trajectories):
                rs = rewards[i][start : ends[i]]
                rs[-1] = scores[i]
                all_rewards[i] = rs

            new_rollout = [
                PPORLElement(
                    query_tensor = query_tensors[i],
                    response_tensor = response_tensors[i],
                    logprobs = truncated_logprobs[i],
                    values = truncated_values[i],
                    rewards = all_rewards[i],
                )
                for i in range(n_trajectories)
            ]

            del query_tensors, response_tensors, truncated_logprobs, truncated_values, all_rewards
            
            all_rollouts += new_rollout

        score = torch.tensor(scores).mean().detach().cpu().item()

        return all_rollouts, score
    
    def logprobs_from_logits(self, logits, labels):
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
        return logprobs_labels.squeeze(-1)
    
    def loss_fn(self, mini_batch):
        query_tensors = mini_batch.query_tensors
        response_tensors = mini_batch.response_tensors
        old_logprobs = mini_batch.logprobs
        old_values = mini_batch.values
        old_rewards = mini_batch.rewards

        response_length = old_rewards.shape[1]

        advantages, returns = self.gae(old_values, old_rewards)

        trajectories = torch.hstack([mini_batch.query_tensors, mini_batch.response_tensors])
        attention_mask = trajectories.not_equal(self.tokenizer.pad_token_id).long()
        logits, values_pred = self.policy_model(trajectories, attention_mask=attention_mask)

        values_pred = values_pred[:, :-1]
        logprobs = self.logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
        attention_mask = attention_mask[:, :-1]

        start = query_tensors.shape[1] - 1
        end = start + response_length
        logprobs, values_pred, mask = (
            logprobs[:, start:end],
            values_pred[:, start:end],
            attention_mask[:, start:end],
        )

        loss = self.ppo_loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )

        return loss, old_rewards[:,-1].mean().item()
    
    def ppo_loss(self, logprobs, values, old_logprobs, old_values, advantages, returns, mask):
        values_clipped = torch.clamp(
            values,
            old_values - config.cliprange_value,
            old_values + config.cliprange_value,
        )
        n = mask.sum()
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.cliprange, 1.0 + config.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
        loss = pg_loss + config.vf_coef * vf_loss
        return loss
    
    def gae(self, values, rewards):
        advantages = torch.zeros_like(rewards, device=rewards.device)
        last_advantage = 0
        last_value = 0
        with torch.no_grad():
            for t in reversed(range(rewards.shape[1])):
                delta = rewards[:, t] + config.gamma * last_value - values[:, t]
                last_advantage = delta + config.gamma * config.lam * last_advantage
                advantages[:, t] = last_advantage
                last_value = values[:, t]
            returns = advantages + values
        return advantages, returns

    def save_trained_model(self, save_dir, model_name):
        self.policy_model.save_model(save_dir, model_name)