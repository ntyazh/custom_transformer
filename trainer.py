import torch
from livelossplot import PlotLosses
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        tokenizer,
        learning_rate=3e-4,
        weight_decay=0.01,
        clip_grad_norm=1.0,
        n_steps=10_000,
        val_every_n_steps=1_000,
        plot_every_n_steps=100,
        n_accumulation_steps=16,
    ):
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.n_steps = n_steps
        self.val_every_n_steps = val_every_n_steps
        self.plot_every_n_steps = plot_every_n_steps
        self.n_accumulation_steps = n_accumulation_steps

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print("running on device", self.device)

    @torch.no_grad()
    def validate(self, model, val_loader):
        model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            val_loss += self.cross_entropy_loss(input_ids, attention_mask, logits)
        return val_loss / len(val_loader)

    def run(self, model, train_loader, val_loader):
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * self.n_steps,
            num_training_steps=self.n_steps,
        )
        model.train()

        plotlosses = PlotLosses(figsize=(14, 5), step_names="Step")
        logs = {"lr": 0}  # , "epoch": 0}

        data_iter = iter(train_loader)
        for iter_num in range(self.n_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                # logs["epoch"] += 1
                batch = next(data_iter)

            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            loss = self.cross_entropy_loss(input_ids, attention_mask, logits)
            loss = loss / self.n_accumulation_steps

            # backprop and update the parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            if iter_num % self.n_accumulation_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if iter_num > 0 and iter_num % self.val_every_n_steps == 0:
                val_loss = self.validate(model, val_loader)
                plotlosses.update({"val_loss": val_loss.item()}, current_step=iter_num)
                plotlosses.send()
                model.train()

            if iter_num % self.plot_every_n_steps == 0:
                logs["loss"] = loss.item()
                logs["lr"] = scheduler.get_last_lr()[0]
                plotlosses.update(logs, current_step=iter_num)
                plotlosses.send()

        val_loss = self.validate(model, val_loader)
        plotlosses.update({"val_loss": val_loss.item()}, current_step=iter_num)
        plotlosses.send()

    def cross_entropy_loss(
        self, input_ids: Tensor, attention_mask: Tensor, logits: Tensor
    ) -> Tensor:
        """Calculate Cross-Entropy loss for Language Modeling task
        Under the hood:
        1. Create targtes based on input ids
        2. Masked out tokens corresponded to paddings
        3. Calculate cross entropy loss

        Args:
            input_ids: tensor with input ids, shape [bs, seq len]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len]
            logits: predicted logits, shape [bs, seq len, vocab size]
        Return:
            cross entropy loss, single-item tensor
        """
        target = input_ids[:, 1:].squeeze()
        logits = logits[:, :-1, :].transpose(1, 2)
        loss = F.cross_entropy(
            logits, target, ignore_index=self.tokenizer.eos_token_id, reduction="none"
        )
        loss = (loss * attention_mask[:, 1:].squeeze()).mean()
        return loss / target.shape[0]  # batch_size


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Scheduler for Optimizer with linear warmup and linear decay to the end of training

    Args:
        optimizer: torch optimizer to control learning rate
        num_warmup_steps: number of warmup steps
        num_training_steps: total number of training steps
    Return:
        torch learning rate scheduler
    """
    assert num_training_steps >= num_warmup_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / num_warmup_steps
        return float(num_training_steps - current_step) / float(
            num_training_steps - num_warmup_steps
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
