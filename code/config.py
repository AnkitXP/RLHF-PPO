class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict({
        'policy_model_name': 'lvwerra/gpt2-imdb',
        'reward_model_name': 'lvwerra/distilbert-imdb',
        'save_dir':'../saved_models/',
        'seq_length': 256,
        'batch_size': 32,
        'lr': 0.00006,
        'prompt_size': 30,
        'prompt_batch_size': 128,
        'num_rollouts': 128,
        'epochs': 100,
        'ppo_epochs': 4,
        'gen_kwargs': {
            'max_new_tokens': 40,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': True
        },
        'kl_coef': 0.01,
        'gamma': 1,
        'lam': 0.95,
        'cliprange': 0.2,
        'cliprange_value': 0.2,
        'vf_coef': 1,
    })