"""
CFM FlowMP Trainer

Main training class that handles:
- Training loop
- Validation
- Logging
- Checkpointing
- Learning rate scheduling
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .flow_matching import FlowMatchingLoss, FlowMatchingConfig, FlowInterpolator


@dataclass
class TrainerConfig:
    """Configuration for CFM trainer."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "warmup_cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = False
    
    # Mixed precision
    use_amp: bool = True
    
    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Flow matching config
    flow_config: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of the model weights that are updated
    with exponential moving average during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        self.step += 1
        
        if self.step % self.update_every != 0:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'step': self.step,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class CFMTrainer:
    """
    Trainer for Conditional Flow Matching with FlowMP architecture.
    
    Handles the complete training pipeline including:
    - Data loading and batching
    - Flow matching interpolation and loss computation
    - Gradient updates with mixed precision
    - Learning rate scheduling
    - EMA weight averaging
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig = None,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        logger: Any = None,  # wandb, tensorboard, etc.
    ):
        """
        Initialize CFM trainer.
        
        Args:
            model: FlowMP transformer model
            config: Trainer configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            logger: Logger for metrics (optional)
        """
        self.config = config or TrainerConfig()
        self.model = model.to(self.config.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        
        # Initialize loss function
        self.loss_fn = FlowMatchingLoss(self.config.flow_config)
        self.interpolator = FlowInterpolator(self.config.flow_config)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        
        # Initialize scheduler (will be set up in train())
        self.scheduler = None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # EMA
        self.ema = None
        if self.config.use_ema:
            self.ema = EMA(
                model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Dictionary containing:
                - 'positions': [B, T, D] target position trajectory
                - 'velocities': [B, T, D] target velocity trajectory
                - 'accelerations': [B, T, D] target acceleration trajectory
                - 'start_pos': [B, D] starting position
                - 'goal_pos': [B, D] goal position
                - 'start_vel': [B, D] starting velocity (optional)
                
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move batch to device
        q_1 = batch['positions'].to(self.config.device)
        q_dot_1 = batch['velocities'].to(self.config.device)
        q_ddot_1 = batch['accelerations'].to(self.config.device)
        start_pos = batch['start_pos'].to(self.config.device)
        goal_pos = batch['goal_pos'].to(self.config.device)
        start_vel = batch.get('start_vel')
        if start_vel is not None:
            start_vel = start_vel.to(self.config.device)
        
        B = q_1.shape[0]
        
        # Sample flow time
        t = self.interpolator.sample_time(B, self.config.device, q_1.dtype)
        
        # Construct interpolated states and targets
        interp_result = self.interpolator.interpolate_trajectory(
            q_1=q_1,
            q_dot_1=q_dot_1,
            q_ddot_1=q_ddot_1,
            t=t,
        )
        
        x_t = interp_result['x_t']
        target = interp_result['target']
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            model_output = self.model(
                x_t=x_t,
                t=t,
                start_pos=start_pos,
                goal_pos=goal_pos,
                start_vel=start_vel,
            )
            
            loss_dict = self.loss_fn(model_output, target)
            loss = loss_dict['loss'] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary of average validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()
        
        total_loss = 0
        total_loss_vel = 0
        total_loss_acc = 0
        total_loss_jerk = 0
        num_batches = 0
        
        for batch in self.val_dataloader:
            q_1 = batch['positions'].to(self.config.device)
            q_dot_1 = batch['velocities'].to(self.config.device)
            q_ddot_1 = batch['accelerations'].to(self.config.device)
            start_pos = batch['start_pos'].to(self.config.device)
            goal_pos = batch['goal_pos'].to(self.config.device)
            start_vel = batch.get('start_vel')
            if start_vel is not None:
                start_vel = start_vel.to(self.config.device)
            
            B = q_1.shape[0]
            
            # Sample multiple time steps for robust evaluation
            t = self.interpolator.sample_time(B, self.config.device, q_1.dtype)
            
            interp_result = self.interpolator.interpolate_trajectory(
                q_1=q_1,
                q_dot_1=q_dot_1,
                q_ddot_1=q_ddot_1,
                t=t,
            )
            
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                model_output = self.model(
                    x_t=interp_result['x_t'],
                    t=t,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    start_vel=start_vel,
                )
                
                loss_dict = self.loss_fn(model_output, interp_result['target'])
            
            total_loss += loss_dict['loss'].item()
            total_loss_vel += loss_dict['loss_vel'].item()
            total_loss_acc += loss_dict['loss_acc'].item()
            total_loss_jerk += loss_dict['loss_jerk'].item()
            num_batches += 1
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        return {
            'val_loss': total_loss / num_batches,
            'val_loss_vel': total_loss_vel / num_batches,
            'val_loss_acc': total_loss_acc / num_batches,
            'val_loss_jerk': total_loss_jerk / num_batches,
        }
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename (default: step_{global_step}.pt)
            is_best: Whether this is the best checkpoint so far
        """
        if filename is None:
            filename = f"step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if checkpoint['ema_state_dict'] and self.ema:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(
        self,
        num_epochs: int = None,
        resume_from: str = None,
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs (overrides config if provided)
            resume_from: Path to checkpoint to resume from
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed from checkpoint: {resume_from}")
            print(f"Starting from step {self.global_step}, epoch {self.current_epoch}")
        
        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        
        # Set up scheduler
        if self.config.scheduler_type == "warmup_cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=self.config.min_lr / self.config.learning_rate,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_steps / total_steps,
            )
        
        print(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
        print(f"Using device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            epoch_steps = 0
            
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['loss']
                epoch_steps += 1
                
                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss_dict['loss']:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    })
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        if self.logger:
                            self.logger.log({
                                'train/loss': loss_dict['loss'],
                                'train/loss_vel': loss_dict['loss_vel'],
                                'train/loss_acc': loss_dict['loss_acc'],
                                'train/loss_jerk': loss_dict['loss_jerk'],
                                'train/lr': self.optimizer.param_groups[0]['lr'],
                                'train/step': self.global_step,
                            })
                    
                    # Validation
                    if self.global_step % self.config.eval_interval == 0:
                        val_metrics = self.validate()
                        
                        if val_metrics:
                            print(f"\nStep {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f}")
                            
                            if self.logger:
                                self.logger.log({
                                    **val_metrics,
                                    'train/step': self.global_step,
                                })
                            
                            # Check for best model
                            if val_metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val_loss']
                                self.save_checkpoint(is_best=True)
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        if not self.config.save_best_only:
                            self.save_checkpoint()
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
            
            # Epoch-level validation
            val_metrics = self.validate()
            if val_metrics:
                print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(filename="final_model.pt")
        print("Training completed!")
    
    def get_ema_model(self) -> nn.Module:
        """Get model with EMA weights applied."""
        if self.ema is not None:
            self.ema.apply_shadow()
            return self.model
        return self.model
