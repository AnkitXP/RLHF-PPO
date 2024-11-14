class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict({
        'policy_model_name': 'lvwerra/gpt2-imdb',
        'reward_model_name': 'siebert/sentiment-roberta-large-english',
        'save_dir':'../saved_models/',
        'seq_length': 128,
        'batch_size': 256,
        'lr': 1.41e-5,
        'prompt_size': 30,
        'prompt_batch_size': 128,
        'num_rollouts': 64,
        'epochs': 20000,
        'ppo_epochs': 4,
        'gen_kwargs': {
            'max_new_tokens': 40,
            'top_k': 50,
            'top_p': 0.95,
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