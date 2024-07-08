from typing import List, Tuple
from datasets import load_dataset, Dataset, IterableDataset
from transformers import PreTrainedTokenizer
import random

from .config import ScriptArguments

def tokenization(batch, tokenizer: PreTrainedTokenizer):
    batch = tokenizer(batch['text'])
    return batch

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

def get_zipped_dataset_generator(datasets: Tuple[Dataset]):
    columns = datasets[0].column_names
    dataset_len = len(datasets[0])
    for ex_i in range(dataset_len):
        yield {
            f'{key}': [ds[ex_i][key] for ds_i, ds in enumerate(datasets)]
            for key in columns
        }

def zip_datasets(datasets: Tuple[Dataset]):
    dataset = IterableDataset.from_generator(
        get_zipped_dataset_generator,
        gen_kwargs={'datasets': datasets}
    ).with_format(type='torch')
    return dataset

def get_datasets(args: ScriptArguments, tokenizer: PreTrainedTokenizer):
    datasets = (load_dataset('roneneldan/TinyStories', split='train') for _ in range(args.n_streams))
    eval_datasets = (load_dataset('roneneldan/TinyStories', split='validation[:384]') for _ in range(args.n_streams))
    datasets = tuple([
            dataset.map(tokenization, batched=True, num_proc=args.num_proc, remove_columns=list(dataset.features), fn_kwargs={'tokenizer': tokenizer})
            .map(group_texts, batched=True, num_proc=args.num_proc, fn_kwargs={'block_size': args.block_size})
            .shuffle(seed=42 + i) # is this really slow? can also convert to iterable and then shuffle
        for i, dataset in enumerate(list(datasets))
    ])
    eval_datasets = tuple([
            dataset.map(tokenization, batched=True, num_proc=args.num_proc, remove_columns=list(dataset.features), fn_kwargs={'tokenizer': tokenizer})
        for i, dataset in enumerate(list(eval_datasets))
    ])
    dataset = zip_datasets(datasets)
    eval_dataset = zip_datasets(eval_datasets)

    return dataset, eval_dataset

# dataset = zip_datasets((dataset_1,))
# eval_dataset = zip_datasets((eval_dataset_1,))
# dataset = dataset_1.map(tokenization, batched=True, num_proc=16).shuffle(seed=42)
# eval_dataset = eval_dataset_1.map(tokenization, batched=True, num_proc=16).shuffle(seed=42)
