from keras.callbacks import Callback
import multiprocessing
import time


def epoch_end_callback(epoch, logs, model, lock, callback_func):
    callback_func(epoch, logs, model, lock)
    

class ParallelFinder:
    def __init__(self, models):
        self.models = models
        manager = multiprocessing.Manager()
        self.logs = manager.dict()
        self.logs['best_loss'] = 1e9
        self.logs['best_time'] = 1e9
        self.loss = manager.list()
        self.time = manager.list()
        self.lock = multiprocessing.Lock()
    
    class FinderCallback(Callback):
        def __init__(self, finder, model):
            super().__init__()
            self.finder = finder
            self.model = model
    
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()
    
        def on_epoch_end(self, epoch, logs, model=None, lock=None):
            lock.acquire()
            time_ = time.time() - self.epoch_start
            loss = logs['loss']
            
            if epoch+1 == self.epochs:
                self.loss.append(loss)
                self.time.append(model.time)
                if loss < self.logs['best_loss']:
                    self.logs['best_loss_model'] = model
                    self.logs['best_loss'] = loss
                    self.logs['time'] = time_
                if model.time < self.logs['best_time']:
                    self.logs['best_time_model'] = model
                    self.logs['best_time'] = time_
                    self.logs['loss'] = loss
            lock.release()

    def find(self, train_ds=None, batch_size=64, epochs=1, **kw_fit):
        self.epochs = epochs

        process_list=[]
        for i in range(len(self.models)):
            callback = ParallelFinder.FinderCallback(self, self.models[i])
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
