import wandb
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Callable
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch import save as torch_save

from app.classifier import Classifier, TinyLLMClassifier
from app.classifier import Tokenizer

@dataclass
class TrainingArgs:
    num_classes: int
    output_dir: Path
    epochs: int
    warmup_ratio: float
    valid_interval: int
    save_ckpt: str | None
    save_best_only: bool
    submit_file: str | None
    device: str
    wandb_project: str
    wandb_run_name: str
    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.0
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    after_step: Callable[[Classifier, int], None] | None = None
    after_epoch: Callable[[Classifier, int], None] | None = None

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss function for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: Model output logits, shape (batch_size, num_classes).
        targets: Ground truth labels, shape (batch_size,).
        alpha: Balancing factor, typically 1.0 or class weights.
        gamma: Focusing parameter, higher values focus more on hard examples.
    
    Returns:
        Loss value.
    """
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Get probability of true class
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Compute log probabilities
    log_p_t = torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Compute focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    loss = -alpha * ((1 - p_t) ** gamma) * log_p_t
    
    return loss.mean()


def cross_entropy_with_label_smoothing(
    base_criterion: Callable,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    """
    Cross-entropy loss function with label smoothing.
    
    Args:
        base_criterion: Base loss function.
        logits: Model output logits, shape (batch_size, num_classes).
        targets: Ground truth labels, shape (batch_size,).
        num_classes: Number of classes.
        label_smoothing: Label smoothing coefficient, range [0, 1).
    
    Returns:
        Loss value.
    """
    if label_smoothing == 0.0:
        return base_criterion(logits, targets)
    
    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Create smoothed label distribution
    # True class probability: 1 - label_smoothing
    # Other classes probability: label_smoothing / (num_classes - 1)
    smooth_targets = torch.zeros_like(log_probs)
    smooth_targets.fill_(label_smoothing / (num_classes - 1))
    smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    
    # Compute cross-entropy: -sum(smooth_targets * log_probs)
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    
    return loss


class Trainer:
    def __init__(
        self,
        *,
        model: Classifier,
        tokenizer: Tokenizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_df: pd.DataFrame,
        optimizer,
        criterion,
        lr_scheduler,
        training_args: TrainingArgs,
    ):
        self.model: Classifier = model
        self.tokenizer: Tokenizer = tokenizer
        self.train_dataloader: DataLoader = train_dataloader
        self.valid_dataloader: DataLoader = valid_dataloader
        self.test_df = test_df
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = training_args

        # Create base loss function based on loss_type
        if self.args.loss_type == "focal_loss":
            self.base_criterion = lambda logits, labels: focal_loss(
                logits, labels, self.args.focal_alpha, self.args.focal_gamma
            )
        else:  # cross_entropy
            self.base_criterion = criterion

        # Apply label smoothing if enabled (only for cross_entropy)
        if self.args.label_smoothing > 0.0 and self.args.loss_type == "cross_entropy":
            self.criterion = lambda logits, labels: cross_entropy_with_label_smoothing(
                self.base_criterion, logits, labels, self.args.num_classes, self.args.label_smoothing
            )
        else:
            self.criterion = self.base_criterion

        self.best_valid_loss = float("inf")
        self.global_step = 0
        self.total_steps = len(self.train_dataloader) * self.args.epochs
        self.warmup_steps = int(self.args.warmup_ratio * self.total_steps)
        self.initial_lrs = [group["lr"] for group in self.optimizer.param_groups]
        
        if self.args.wandb_run_name:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name, config=vars(self.args))

    def _apply_warmup(self):
        if self.lr_scheduler:
            return self.lr_scheduler.update(self.global_step)
        if self.warmup_steps == 0:
            return None
        scale = min(1.0, self.global_step / self.warmup_steps)
        for base_lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale
        return None

    def validate(self):
        valid_loss = 0.0
        correct = 0
        counter = [0] * self.args.num_classes
        all_labels: list[int] = []
        all_predicts: list[int] = []
        self.model.eval()
        for batch in tqdm(self.valid_dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"]
            lengths = batch["lengths"]
            labels = batch["labels"]
            logits = self.model(input_ids, lengths)
            # Use base criterion for validation (without label smoothing)
            loss = self.base_criterion(logits, labels)
            valid_loss += loss.item()
            predict = logits.argmax(dim=-1)
            correct += (predict == labels).sum().item()
            for i in range(self.args.num_classes):
                counter[i] += (predict == i).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_predicts.extend(predict.cpu().tolist())
        valid_loss /= len(self.valid_dataloader)
        print(f"Validation Loss: {valid_loss:.4f}")
        total_valid = len(self.valid_dataloader.dataset)
        print(f"Validation Accuracy: {correct / total_valid:.4f}")
        print(f"Distribution of predictions: {counter}")
        print("Classification Report:")
        print(classification_report(all_labels, all_predicts))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_predicts))
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            if self.args.save_ckpt is not None and self.args.save_best_only:
                print(f"New best model found, saving checkpoint to {self.args.save_ckpt}...")
                torch_save(self.model.state_dict(), self.args.output_dir / self.args.save_ckpt)
        if self.args.wandb_run_name:
            wandb.log({
                "valid/loss": valid_loss,
                "valid/accuracy": correct / len(self.valid_dataloader.dataset),
            }, step=self.global_step)
        return valid_loss

    def train(self):
        print(f"Training on device: {self.args.device}")
        print("Architecture: ")
        print(self.model)
        for epoch in range(self.args.epochs):
            print(f"Starting epoch {epoch + 1}/{self.args.epochs}...")
            self.model.train()

            ema_loss = 0.0
            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}") as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    if idx % self.args.valid_interval == 0:
                        self.validate()
                        self.model.train()
                    self._apply_warmup()
                    input_ids = batch["input_ids"]
                    lengths = batch["lengths"]
                    labels = batch["labels"]
                    logits = self.model(input_ids, lengths)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1
                    
                    if self.args.after_step is not None:
                        self.args.after_step(self.model, self.global_step)
                    
                    ema_loss = 0.98 * ema_loss + 0.02 * loss.item() if ema_loss != 0.0 else loss.item()
                    avg_lr = sum(group["lr"] for group in self.optimizer.param_groups) / len(self.optimizer.param_groups)
                    pbar.set_postfix({"loss": f"{ema_loss:.3f}", "lr": f"{avg_lr:.2e}"})
                    pbar.update(1)
                    if self.args.wandb_run_name:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": avg_lr,
                        }, step=self.global_step)
            
            if self.args.after_epoch is not None:
                self.args.after_epoch(self.model, epoch)

        if self.args.save_ckpt is not None and not self.args.save_best_only:
            torch_save(self.model.state_dict(), self.args.output_dir / self.args.save_ckpt)

    def predict_test(self):
        if self.args.submit_file is None:
            return
        print("Predicting on test set...")
        phrases = self.test_df["Phrase"].tolist()
        test_batchsize = 32
        predictions: list[int] = []
        for i in tqdm(range(0, len(phrases), test_batchsize), desc="Testing"):
            batch_start = i
            batch_end = min(i + test_batchsize, len(phrases))
            batch_phrases = phrases[batch_start:batch_end]
            batch_predictions = self.model.predict(batch_phrases, self.tokenizer)
            predictions.extend(batch_predictions)
        submission = pd.DataFrame({"PhraseId": self.test_df["PhraseId"], "Sentiment": predictions})
        submission.to_csv(self.args.output_dir / self.args.submit_file, index=False)
