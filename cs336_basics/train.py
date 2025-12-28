import torch
import torch.nn as nn
import numpy as np
import os
import time
from typing import Callable
from cs336_basics.models import *
from cs336_basics.losses import *
from cs336_basics.optimizer import *
from cs336_basics.utils import *

class Trainer:
    def __init__(
        self,
        config: dict,
        load_from: str | os.PathLike | None = None,
    ):
        self.config = config
        self.model: nn.Module = build_model(config)
        self.optimizer: torch.optim.Optimizer = build_optimizer(config, self.model)
        self.loss_fn: Callable = cross_entropy_loss
        self.print_every = 10
        self.valid_every = 100

        if load_from is not None:
            load_checkpoint(load_from, self.model, self.optimizer)

        os.makedirs(self.config["ckpt"]["dir"], exist_ok=True)
        self.train_data, self.valid_data = self.load_data()

    def load_data(self):
        """Set up memory-efficient dataloading."""
        train_path = self.config["data"]["train_data"]["path"]
        valid_path = self.config["data"]["valid_data"]["path"]
        train_data = np.load(train_path, mmap_mode='r', allow_pickle=True).astype(np.uint16)
        valid_data = np.load(valid_path, mmap_mode='r', allow_pickle=True).astype(np.uint16)
        print('loaded data')
        assert np.max(self.valid_data) <= np.iinfo(np.uint16).max
        return train_data, valid_data

    def train(self):
        hyperparams = self.config["hyperparams"]
        for i in range(hyperparams["num_iterations"]):
            iter_start = time.time()
            batch, targets = get_batch(
                self.train_data,
                hyperparams["batch_size"],
                hyperparams["context_length"],
                hyperparams["device"],
                hyperparams["dtype"],
                hyperparams["randomize"]
            )
            lr = lr_cosine_schedule(
                i, 
                hyperparams["max_learning_rate"], 
                hyperparams["min_learning_rate"], 
                hyperparams["warmup_iters"], 
                hyperparams["cosine_cycle_iters"]
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            logits = self.model(batch)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            gradient_clipping(self.model.parameters(), hyperparams["max_l2_norm"])
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % self.print_every == 0:
                print(f"Iteration {i} | Loss {loss.item():.4f} | Time {time.time() - iter_start:.2f}s")

            if i % self.config["ckpt"]["save_every"] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    i,
                    os.path.join(self.config["ckpt"]["dir"], f"checkpoint_{i}.pt")
                )
            
            if i % self.valid_every == 0:
                self.evaluate()
        
        save_checkpoint(
            self.model,
            self.optimizer,
            hyperparams["num_iterations"] - 1,
            os.path.join(self.config["ckpt"]["dir"], "last.pt")
        )

    def evaluate(self):
        with torch.no_grad():
            hyperparams = self.config["hyperparams"]
            batch, targets = get_batch(
                self.valid_data,
                hyperparams["batch_size"],
                hyperparams["context_length"],
                hyperparams["device"],
                hyperparams["dtype"],
                randomize=False
            )
            logits = self.model(batch)
            loss = self.loss_fn(logits, targets)
            print(f"Validation Loss {loss.item():.4f}")
def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
def main():
    config_dir = "/root/cs336/assignment1-basics/cs336_basics/configs"
    config = load_config(os.path.join(config_dir, "config.yaml"))
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()



