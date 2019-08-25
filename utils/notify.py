# Importing packages and libraries
import keras 
from messageUtils import send_notification

class Notify(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        accuracy = logs.get('acc')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        message = "Epoch: " + str(self.epoch) + "Loss: " + str(loss)[0:5] + "Val.Loss: " + str(val_loss)[0:5] + "Acc: " + str(accuracy)[0:5] + "Val.Acc: " + str(val_acc)[0:5]
        send_notification(message)
        self.epoch += 1