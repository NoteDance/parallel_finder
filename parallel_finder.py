import multiprocessing
import time
from tensorflow import keras


class FinderCallback(keras.callbacks.Callback):
    def __init__(self, finder, model_idx):
        super().__init__()
        self.finder = finder
        self.model_idx = model_idx
        self.lock = finder.lock

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_elapsed = time.time() - self.epoch_start
        loss = logs.get('loss', None)

        if epoch + 1 == self.finder.epochs:
            with self.lock:
                self.finder.logs[f'model_{self.model_idx}_loss'] = loss
                self.finder.logs[f'model_{self.model_idx}_time'] = time_elapsed

                if loss < self.finder.logs['best_loss']:
                    self.finder.logs['best_loss'] = loss
                    self.finder.logs['best_loss_model_idx'] = self.model_idx
                    self.finder.logs['time_for_best_loss'] = time_elapsed

                if time_elapsed < self.finder.logs['best_time']:
                    self.finder.logs['best_time'] = time_elapsed
                    self.finder.logs['best_time_model_idx'] = self.model_idx
                    self.finder.logs['loss_for_best_time'] = loss


class ParallelFinder:
    def __init__(self, model_fn_list):
        self.model_fn_list = model_fn_list
        manager = multiprocessing.Manager()
        self.logs = manager.dict()
        self.logs['best_loss'] = float('inf')
        self.logs['best_loss_model_idx'] = None
        self.logs['best_time'] = float('inf')
        self.logs['best_time_model_idx'] = None
        self.lock = multiprocessing.Lock()

    def _train_single(self, idx, train_data, train_labels, epochs, batch_size, **kw_fit):
        model = self.model_fn_list[idx]()
        self.epochs = epochs
        cb = FinderCallback(self, idx)

        model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[cb],
            **kw_fit
        )

    def find(self, train_data, train_labels, epochs=1, batch_size=32, **kw_fit):
        processes = []
        for idx in range(len(self.model_fn_list)):
            p = multiprocessing.Process(
                target=self._train_single,
                args=(idx, train_data, train_labels, epochs, batch_size),
                kwargs=kw_fit
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        return
