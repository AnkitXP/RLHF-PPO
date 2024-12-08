class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict({
        'policy_model_name': 'EKKam/opt-1.5b_imdb_sft',
        'reward_model_name': 'siebert/sentiment-roberta-large-english',
        'save_dir':'/saved_models/',
        'seq_length': 1024,
        'batch_size': 64,
        'lr': 6e-5,
        'prompt_size': 30,
        'prompt_batch_size': 128,
        'num_rollouts': 128,
        'epochs': 10000,
        'ppo_epochs': 4,
        'gen_kwargs': {
            'max_new_tokens': 128,
            'top_k': 0.0,
            'top_p': 1.0,
            'do_sample': True,
            'temperature': 0.7
        },
        'kl_coef': 0.2,
        'gamma': 1,
        'lam': 0.95,
        'cliprange': 0.2,
        'cliprange_value': 0.2,
        'vf_coef': .1,
    })