from dataclasses import dataclass
from typing import Callable, Literal

import torch as t
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from part2_cnns.solutions import Linear, ResNet34
from part3_optimization.solutions import WandbResNetFinetuningArgs, get_cifar

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()

def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)  # Send tensor to CPU before sending
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the CPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank == src:
        #print(f"{rank=}, sending {tensor=}")
        for i in range(world_size):
            if i != src:
                dist.send(tensor=tensor, dst=i)
    else:
        received_tensor = t.zeros_like(tensor)
        dist.recv(received_tensor, src=src)
        tensor.copy_(received_tensor)
    #raise NotImplementedError()

def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank != dst:
        dist.send(tensor, dst)
    
    else:
        for i in range(world_size):
            if i != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, i)
                tensor += received_tensor
    
        if op == "mean":
            tensor /= world_size
    #raise NotImplementedError()


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, 0, op)
    broadcast(tensor, rank, world_size, 0)
    #raise NotImplementedError()

class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs
    #examples_seen: int = 0

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        for param in self.model.parameters():
            broadcast(param.data, self.rank, self.args.world_size)
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.trainset, self.testset = get_cifar()

        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size,
            rank=self.rank
        )
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, sampler=self.train_sampler, num_workers=2, pin_memory=True)
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset,
            num_replicas=self.args.world_size,
            rank=self.rank
        )
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, sampler=self.test_sampler, num_workers=2, pin_memory=True)
        self.examples_seen = 0
        #raise NotImplementedError()

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        for params in self.model.parameters():
            all_reduce(params.grad, self.rank, self.args.world_size)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.examples_seen += imgs.shape[0] * self.args.world_size

        return loss
        raise NotImplementedError()

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples = len(imgs)

        tensor = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size)
        total_correct, total_samples = tensor.tolist()

        accuracy = total_correct / total_samples
        
        return accuracy
        raise NotImplementedError()

    def train(self):
        self.pre_training_setup()

        accuracy = self.evaluate()  # our evaluate method is the same as parent class

        for epoch in range(self.args.epochs):

            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen=:06}")

            accuracy = self.evaluate()

            if self.rank == 0:
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    accuracy=f"{accuracy:.3f}",
                    ex_seen=f"{self.examples_seen=:06}",
                )

        if self.rank == 0:
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")
        
        #raise NotImplementedError()


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()