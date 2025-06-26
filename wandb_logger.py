"""
Weights & Biases Logger for VideoMAE
Integrates with metric_logger to provide real-time training progress logging
"""

import wandb
import utils


class WandbLogger:
    """
    A wrapper around wandb that integrates with VideoMAE's metric_logger
    to provide comprehensive training progress logging.
    """
    
    def __init__(self, args, model=None):
        """
        Initialize wandb logger
        
        Args:
            args: Training arguments containing wandb config
            model: Model to optionally track gradients and parameters
        """
        self.args = args
        self.enabled = args.use_wandb and utils.is_main_process()
        self.model = model
        self.step_count = 0
        
        if self.enabled:
            # Auto-generate run name if not provided
            if args.wandb_run_name is None:
                args.wandb_run_name = f"{args.model}_{args.data_set}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
            
            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                resume='allow'
            )
            
            # Watch model for gradients and parameters (optional)
            if model is not None:
                wandb.watch(model, log='all', log_freq=100)
            
            print(f"✅ Weights & Biases initialized: {wandb.run.url}")
    
    def log_training_step(self, metric_logger, epoch, step, global_step=None):
        """
        Log training metrics from metric_logger to wandb
        
        Args:
            metric_logger: The VideoMAE MetricLogger instance
            epoch: Current epoch
            step: Current step within epoch  
            global_step: Global training step (optional)
        """
        if not self.enabled:
            return
        
        # Extract metrics from metric_logger
        metrics = {}
        
        # Get all metrics from metric_logger
        for name, meter in metric_logger.meters.items():
            if meter.count > 0:  # Only log if we have data
                metrics[f'train/{name}'] = meter.global_avg
                
                # Also log current value for some key metrics
                if name in ['loss', 'class_acc', 'lr']:
                    metrics[f'train/{name}_current'] = meter.value
        
        # Add epoch and step information
        metrics['epoch'] = epoch
        metrics['step'] = step
        
        # Always use provided global_step when available, ensure monotonic steps
        if global_step is not None:
            log_step = global_step
            # Force step_count to be monotonically increasing
            self.step_count = max(self.step_count, global_step + 1)
        else:
            log_step = self.step_count
            self.step_count += 1
        
        # Add GPU memory usage if available
        if hasattr(metric_logger, '_get_memory_usage'):
            metrics['system/gpu_memory_mb'] = metric_logger._get_memory_usage()
        
        wandb.log(metrics, step=log_step)
    
    def log_validation(self, val_stats, epoch, max_accuracy=None):
        """
        Log validation metrics to wandb
        
        Args:
            val_stats: Dictionary of validation statistics
            epoch: Current epoch
            max_accuracy: Best accuracy so far (optional)
        """
        if not self.enabled:
            return
        
        metrics = {}
        for k, v in val_stats.items():
            metrics[f'val/{k}'] = v
        
        metrics['epoch'] = epoch
        
        if max_accuracy is not None:
            metrics['val/max_accuracy'] = max_accuracy
        
        # Use self.step_count to ensure monotonically increasing steps
        wandb.log(metrics, step=self.step_count)
        self.step_count += 1
    
    def log_epoch_summary(self, train_stats, val_stats=None, epoch=None, max_accuracy=None):
        """
        Log end-of-epoch summary
        
        Args:
            train_stats: Training statistics for the epoch
            val_stats: Validation statistics (optional)
            epoch: Epoch number
            max_accuracy: Best accuracy achieved so far
        """
        if not self.enabled:
            return
        
        metrics = {}
        
        # Log training stats
        for k, v in train_stats.items():
            metrics[f'epoch_summary/train_{k}'] = v
        
        # Log validation stats if available
        if val_stats:
            for k, v in val_stats.items():
                metrics[f'epoch_summary/val_{k}'] = v
        
        if epoch is not None:
            metrics['epoch'] = epoch
            
        if max_accuracy is not None:
            metrics['epoch_summary/max_accuracy'] = max_accuracy
        
        wandb.log(metrics, step=epoch if epoch is not None else self.step_count)
    
    def log_final_results(self, final_top1, final_top5):
        """
        Log final test results
        
        Args:
            final_top1: Final top-1 accuracy
            final_top5: Final top-5 accuracy
        """
        if not self.enabled:
            return
        
        wandb.log({
            'test/final_top1': final_top1,
            'test/final_top5': final_top5,
        })
    
    def log_custom(self, metrics, step=None):
        """
        Log custom metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if not self.enabled:
            return
        
        log_step = step if step is not None else self.step_count
        wandb.log(metrics, step=log_step)
    
    def finish(self):
        """
        Finish the wandb run
        """
        if self.enabled:
            wandb.finish()


class WandbMetricLogger(utils.MetricLogger):
    """
    Extended MetricLogger that automatically logs to wandb
    """
    
    def __init__(self, wandb_logger=None, log_freq=50, delimiter="\t"):
        """
        Initialize the extended metric logger
        
        Args:
            wandb_logger: WandbLogger instance
            log_freq: How often to log to wandb (every N steps)
            delimiter: Delimiter for console output
        """
        super().__init__(delimiter=delimiter)
        self.wandb_logger = wandb_logger
        self.log_freq = log_freq
        self.step_count = 0
        self.epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch"""
        self.epoch = epoch
    
    def log_every(self, iterable, print_freq, header=None):
        """
        Enhanced log_every that also logs to wandb
        """
        for i, obj in enumerate(super().log_every(iterable, print_freq, header)):
            # Log to wandb at specified frequency
            if (self.wandb_logger and 
                self.wandb_logger.enabled and 
                i % self.log_freq == 0):
                self.wandb_logger.log_training_step(
                    self, 
                    self.epoch, 
                    i, 
                    global_step=self.step_count
                )
            
            self.step_count += 1
            yield obj 