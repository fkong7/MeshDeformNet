#Copyright (C) 2021 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import models


class SaveModelOnCD(callbacks.Callback):
    def __init__(self, keys, model_save_path, patience):
        self.keys = keys
        self.save_path = model_save_path
        self.no_improve = 0
        self.patience = patience
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        CD_val_loss = 0.
        for key in self.keys:
            CD_val_loss += logs.get('val_'+key+'_point_loss_cf')
        if CD_val_loss < self.best:
            print("Validation loss decreased from %f to %f, saving model to %s.\n" % (self.best, CD_val_loss, self.save_path))
            self.best = CD_val_loss
            if isinstance(self.model.layers[-2], models.Model):
                self.model.layers[-2].save_weights(self.save_path)
            else:
                self.model.save_weights(self.save_path)
            self.no_improve = 0
        else:
            print("Validation loss did not improve from %f.\n" % self.best)
            self.no_improve += 1
        if self.no_improve > self.patience:
            self.model.stop_training = True


