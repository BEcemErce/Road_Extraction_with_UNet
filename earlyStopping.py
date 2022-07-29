# -*- coding: utf-8 -*-
"""
"""

class EarlyStopping():
    def __init__(self, tolerance=10, min_delta=0.01, delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.delta=delta

    def __call__(self, train_loss, validation_loss):
        score = -validation_loss

        if self.best_score is None:
            self.best_score = score       
        
        elif score < self.best_score + self.delta or (validation_loss - train_loss) > self.min_delta :
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
