import logging as logging_std_library

import a2.training.benchmarks as timer
import torch
import transformers
from transformers.trainer import *  # Ugly but probably needed ... # noeq


class TimerCallback(transformers.TrainerCallback):
    """
    # Example usage:
    tmr = timer.CPUGPUTimer()
    trainer = transformers.Trainer(
        # Other args...
        callbacks=[TimerCallback(tmr, gpu=True)]
    )
    trainer.train()
    tmr.print_all_time_stats()
    """

    def __init__(self, timer, gpu=False):
        super().__init__()
        self.timer = timer
        self.gpu = gpu

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.timer.start(timer.TimeType.EPOCH)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.timer.end(timer.TimeType.EPOCH)
        self.timer.complete_all()

    def on_step_begin(self, args, state, control, **kwargs):
        logging_std_library.info(
            f"Epoch {int(state.epoch)}: Start iteration step {state.global_step}/{state.max_steps} of training..."
        )
        self.timer.start(timer.TimeType.BATCH, gpu=self.gpu)

    def on_step_end(self, args, state, control, **kwargs):
        self.timer.end(timer.TimeType.BATCH, gpu=self.gpu)
        if state.global_step % 10 == 0:
            self.timer.complete_all()


class TimeLoaderWrapper:
    """Wrapper around a DataLoader (*not* a Dataset!) for I/O timing."""

    def __init__(self, loader, timer):
        self.loader = loader
        self.tmr = timer

    @staticmethod
    def time_loader(loader, tmr):
        if len(loader) > 0:
            tmr.start(timer.TimeType.IO)
        for i, data in enumerate(loader):
            tmr.end(timer.TimeType.IO)
            if i % 10 == 0:
                tmr.complete_all()
            yield data
            if i != len(loader) - 1:
                tmr.start(timer.TimeType.IO)

    def __iter__(self):
        return TimeLoaderWrapper.time_loader(self.loader, self.tmr)

    def __len__(self):
        return len(self.loader)

    def reset(self):
        if hasattr(self.loader, "reset"):
            self.loader.reset()


class TrainerWithTimer(Trainer):
    """
    Custom Trainer subclass to support finer-grained timing.
    Should also use the callback above.
    Adapted from original:
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/trainer.py
    """

    def __init__(self, *args, **kwargs):
        [self.timer_callback] = kwargs["callbacks"]
        self.tmr = self.timer_callback.timer
        self.tmr_gpu = self.timer_callback.gpu
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.tmr:
            self.tmr.start(timer.TimeType.FORWARD, gpu=self.tmr_gpu)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        if self.tmr:
            self.tmr.end(timer.TimeType.FORWARD, gpu=self.tmr_gpu)
            self.tmr.start(timer.TimeType.BACKWARD, gpu=self.tmr_gpu)

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.tmr:
            self.tmr.end(timer.TimeType.BACKWARD, gpu=self.tmr_gpu)

        return loss.detach()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            dl = DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            train_sampler = self._get_train_sampler()

            dl = DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=train_sampler,
                collate_fn=data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )

        if self.tmr:
            return TimeLoaderWrapper(dl, self.tmr)
        return dl

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        self.tmr.start(timer.TimeType.SAVING_MODEL)
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        self.tmr.end(timer.TimeType.SAVING_MODEL)
