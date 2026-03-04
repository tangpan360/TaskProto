
class EarlyStopping:
    def __init__(self, patience = 5, min_delta=0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def should_stop(self, epoch_loss, epoch):
        """
        检查是否应该早停
        
        Returns:
            tuple: (should_stop: bool, is_best: bool)
                - should_stop: 是否应该停止训练
                - is_best: 当前是否是最优模型（用于保存权重）
        """
        is_best = False
        
        if epoch == 0 or (epoch_loss < self.best_loss - self.min_delta):
            self.best_loss = epoch_loss
            self.no_improvement_count = 0
            is_best = True
            should_stop = False
        else:
            self.no_improvement_count += 1
            should_stop = self.no_improvement_count >= self.patience
        
        return should_stop, is_best