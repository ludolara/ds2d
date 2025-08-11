import random
from trl import GRPOTrainer
from transformers import TrainerCallback

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, eval_sample_size=50, random_eval=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size
        self.random_eval = random_eval
        
    def get_eval_dataloader(self, eval_dataset=None):
        '''
        Samples the evaluation dataset and returns a subset 
        of size self.eval_sample_size. If random_eval is True, uses different 
        examples each evaluation. If False, uses fixed subset.
        '''
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if len(eval_dataset) <= self.eval_sample_size:
            return super().get_eval_dataloader(eval_dataset)
            
        if self.random_eval:
            idxs = random.sample(range(len(eval_dataset)), self.eval_sample_size)
            eval_subset = eval_dataset.select(idxs)
        else:
            eval_subset = eval_dataset.select(range(self.eval_sample_size))
        return super().get_eval_dataloader(eval_subset)

class BestRewardCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3):
        self.best_reward = None
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_counter = 0
        
    def on_log(self, args, state, control, logs, **kwargs):
        """
        Called after all metrics are computed and logged.
        This fires after evaluation is complete with all metrics available.
        """
        if any(key.startswith('eval_') for key in logs.keys()):
            if "eval_reward" in logs:
                reward = logs["eval_reward"]
                
                # Custom best model tracking and saving
                is_best = False
                if self.best_reward is None or reward > self.best_reward:
                    is_best = True
                    # old_best = self.best_reward
                    self.best_reward = reward
                    
                    # Reset early stopping counter on improvement
                    self.early_stopping_patience_counter = 0
                    
                    # Trigger model save
                    control.should_save = True
                    
                    # if old_best is None:
                    #     print(f"🎯 First evaluation - setting baseline: {reward}")
                    # else:
                    #     print(f"🌟 NEW BEST! {old_best:.6f} → {reward:.6f} - SAVING MODEL!")
                    #     print(f"🔄 Early stopping counter reset to 0")
                else:
                    # No improvement - increment early stopping counter
                    self.early_stopping_patience_counter += 1
                    # print(f"🏆 Current best remains: {self.best_reward:.6f} (current: {reward:.6f})")
                    # print(f"⏳ No improvement for {self.early_stopping_patience_counter}/{self.early_stopping_patience} evaluations")
                    
                    # Check for early stopping
                    if self.early_stopping_patience_counter >= self.early_stopping_patience:
                        # print(f"🛑 EARLY STOPPING TRIGGERED! No improvement for {self.early_stopping_patience} evaluations")
                        control.should_training_stop = True
                
                # Also update trainer's state for consistency
                if is_best:
                    state.best_metric = reward
                    state.best_model_checkpoint = args.output_dir
