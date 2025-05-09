from keras.callbacks import LambdaCallback
import multiprocessing
from functools import partial


def epoch_end_callback(epoch, logs, model, lock, callback_func):
    callback_func(epoch, logs, model, lock)
    

class ModelFinder:
    def __init__(self, models):
        self.models = models
        manager = multiprocessing.Manager()
        self.logs = manager.dict()
        self.logs['best_loss'] = 1e9
        self.lock = multiprocessing.Lock()

    def on_epoch_end(self, epoch, logs, model=None, lock=None):
        lock.acquire()
        loss = logs['loss']
        
        if epoch+1 == self.epochs:
            if loss < self.logs['best_loss']:
                self.logs['best_opt'] = model.optimizer
                self.logs['best_loss'] = loss
        lock.release()

    def find(self, train_ds=None, batch_size=64, epochs=1, **kw_fit):
        self.epochs = epochs

        process_list=[]
        for i in range(len(self.models)):
            partial_callback = partial(
                epoch_end_callback,
                model=self.models[i],
                lock=self.lock,
                callback_func=self.on_epoch_end
            )
            callback = LambdaCallback(on_epoch_end=partial_callback)
            process=multiprocessing.Process(target=self.models[i].fit,kwargs={
                                                    'x': train_ds,
                                                    'epochs': epochs,
                                                    'callbacks': [callback],
                                                    'verbose': 0,
                                                    **kw_fit
                                                })
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()