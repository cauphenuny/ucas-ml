import wandb
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
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
    freeze_base_model: bool
    release_steps: int
    valid_interval: int
    save_ckpt: str | None
    save_best_only: bool
    submit_file: str | None
    use_tinyllm: bool
    device: str
    wandb_project: str
    wandb_run_name: str

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
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.args = training_args

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

    def freeze(self):
        if self.args.use_tinyllm:
            for param in self.model.model.parameters():
                param.requires_grad = False

    def release(self):
        if self.args.use_tinyllm:
            for param in self.model.model.parameters():
                param.requires_grad = True

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
            loss = self.criterion(logits, labels)
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
        if self.args.freeze_base_model:
            assert isinstance(self.model, TinyLLMClassifier)
            self.freeze()

        print(f"Training on device: {self.args.device}")
        for epoch in range(self.args.epochs):
            print(f"Starting epoch {epoch + 1}/{self.args.epochs}...")
            self.model.train()

            ema_loss = 0.0
            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}") as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    if idx % self.args.valid_interval == 0:
                        self.validate()
                        self.model.train()
                    input_ids = batch["input_ids"]
                    lengths = batch["lengths"]
                    labels = batch["labels"]
                    logits = self.model(input_ids, lengths)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1
                    self._apply_warmup()
                    ema_loss = 0.98 * ema_loss + 0.02 * loss.item() if ema_loss != 0.0 else loss.item()
                    avg_lr = sum(group["lr"] for group in self.optimizer.param_groups) / len(self.optimizer.param_groups)
                    pbar.set_postfix({"loss": f"{ema_loss:.3f}", "lr": f"{avg_lr:.2e}"})
                    pbar.update(1)
                    if self.global_step == self.args.release_steps and self.args.freeze_base_model:
                        assert isinstance(self.model, TinyLLMClassifier)
                        print("Releasing base model parameters...")
                        self.release()
                    if self.args.wandb_run_name:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": avg_lr,
                        }, step=self.global_step)

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
