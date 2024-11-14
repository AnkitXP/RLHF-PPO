from datasets import load_dataset

class IMDBDataset:
    def __init__(self, tokenizer, max_length=30):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, split='train'):
        dataset = load_dataset('imdb', split=split)
        dataset = dataset.rename_columns({'text': 'review'})
        dataset = dataset.filter(lambda x: len(x["review"])>200, batched=False)
        dataset = dataset.map(self.tokenize_data, batched=True)
        return dataset['input_ids']

    def tokenize_data(self, data):
        return self.tokenizer(
            data['review'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
