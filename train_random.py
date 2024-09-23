import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import get_peft_model_state_dict

from transformers import Trainer
from torch.utils.data import SequentialSampler,DistributedSampler

from typing import List, Iterator, Optional
import torch.nn as nn
import torch
import math
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler, RandomSampler

from llmzoo.datasets.datasets import make_supervised_data_module
from llmzoo.models import build_model
from llmzoo.utils import safe_save_model_for_hf_trainer


class DistributedSubsetRandomSampler(Sampler):
    def __init__(self, indices, num_replicas=None, rank=None):
        self.indices = indices

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Require distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Require distributed package to be available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.indices) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Generate a list of indices from the random subset
        indices = self.indices[:]

        # Add extra indices to make it evenly divisible
        indices += indices[:self.total_size - len(indices)]

        assert len(indices) == self.total_size

        # Subsample the indices
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class MultiSubsetBatchSampler(Sampler[List[int]]):
    """
    Sample batches within each dataset.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        weights=None
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = True  # For accelerate.BatchSamplerShard
        generator = generator or torch.Generator()  # this might be not necessary
        # virtual sampler for generator synchronizing
        self.sampler = RandomSampler([], num_samples=1, generator=generator)

        # create subset batch_samplers
        subset_indices, start = list(), 0
        for ds in datasets:
            subset_indices.append(list(range(start, start + len(ds))))
            start = sum(len(s) for s in subset_indices)
        # self.batch_samplers = [
        #     BatchSampler(  # TODO: Custom BatchSampler for safe random hard negtives?
        #         SubsetRandomSampler(indices, generator),  # use shared generator
        #         batch_size=batch_size,  # different batch sizes?
        #         drop_last=True  # drop last batch, otherwise things become complicated
        #     )
        #     for indices in subset_indices
        # ]
        self.batch_samplers = [
            BatchSampler(
                DistributedSubsetRandomSampler(indices),
                batch_size=batch_size,  # different batch sizes?
                drop_last=True  # drop last batch, otherwise things become complicated
            )
            for indices in subset_indices
        ]

        # total batch num
        self._total_length = 0
        # batch to task mapping, used when `weights` is `None` to make sure
        # that every batch of each dataset is sampled.
        # This is more reliable than weights by sizes (I think...)
        self._batch_index_to_task = list()
        for i, bs in enumerate(self.batch_samplers):
            self._total_length += len(bs)
            self._batch_index_to_task.extend([i for _ in range(len(bs))])

        if weights is None:
            self.weights = None
        else:
            assert len(weights) == len(self.batch_samplers)
            self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self) -> Iterator[List[int]]:
        bs_iters = [iter(bs) for bs in self.batch_samplers]
        if self.weights is None:
            task_ids = [
                self._batch_index_to_task[i]
                for i in torch.randperm(self._total_length).tolist()
            ]
        else:
            
            task_ids = torch.multinomial(self.weights, self._total_length, True).tolist()
        for tid in task_ids:
            try:
                batch = next(bs_iters[tid])
                yield batch
            except StopIteration:
                bs_iters[tid] = iter(self.batch_samplers[tid])

    def __len__(self) -> int:
        return self._total_length


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class MultiTrainer(Trainer):
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # this dataset is not an IterableDataset
        
        if hasattr(self.train_dataset, 'datasets'):
            train_sampler = DistributedSampler(self.train_dataset)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                # num_workers=self.args.world_size
            )

        train_sampler = DistributedSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.world_size
        )
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path_dir: str=field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = build_model(model_args, training_args)


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = MultiTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if model_args.lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
        if torch.__version__ >= "2":
            model = torch.compile(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    if model_args.lora:
        model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "lora"))
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
