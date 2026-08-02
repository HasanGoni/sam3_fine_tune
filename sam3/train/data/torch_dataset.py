# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset


class GradientAccumulationWrapper:
    """Wrapper to collect multiple batches for gradient accumulation."""
    def __init__(self, dataloader: Iterable, accumulation_steps: int):
        self.dataloader = dataloader
        self.accumulation_steps = accumulation_steps
    
    def __iter__(self):
        batch_list = []
        for batch in self.dataloader:
            batch_list.append(batch)
            if len(batch_list) == self.accumulation_steps:
                yield batch_list
                batch_list = []
        # Yield remaining batches if any (shouldn't happen if drop_last=True)
        if len(batch_list) > 0:
            yield batch_list
    
    def __len__(self):
        # Adjust length for gradient accumulation
        base_len = len(self.dataloader)
        return base_len // self.accumulation_steps


class TorchDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        enable_distributed_sampler=True,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        assert not isinstance(self.dataset, IterableDataset), "Not supported yet"
        if enable_distributed_sampler:
            self.sampler = DistributedSampler(self.dataset, shuffle=self.shuffle)
        else:
            self.sampler = None

    def get_loader(self, epoch) -> Iterable:
        if self.sampler:
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
        
        # Wrap dataloader for gradient accumulation if needed
        if self.gradient_accumulation_steps > 1:
            return GradientAccumulationWrapper(dataloader, self.gradient_accumulation_steps)
        return dataloader
