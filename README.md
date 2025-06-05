# ParallelFinder

A lightweight utility for training multiple Keras models in parallel and comparing their final loss and last-epoch time.

## Overview

* **ParallelFinder**

  * Accepts a list of zero-argument functions, each returning a compiled `keras.Model`.
  * Spawns one `multiprocessing.Process` per model to train simultaneously.
  * Uses a shared, process-safe dictionary (`logs`) to record per-model and global “best” metrics.

* **FinderCallback**

  * Measures the duration of each epoch.
  * On the final epoch, writes the model’s final loss and epoch time into `ParallelFinder.logs`.
  * Updates global best-loss and best-time entries if surpassed.

---

## Quickstart

1. **Define Model Constructors**

   Each function should return a compiled Keras model:

   ```python
   from tensorflow import keras

   def build_dense():
       model = keras.Sequential([
           keras.Input(shape=(784,)),
           keras.layers.Dense(128, activation="relu"),
           keras.layers.Dense(10, activation="softmax"),
       ])
       model.compile(optimizer="adam", loss="categorical_crossentropy")
       return model

   def build_cnn():
       model = keras.Sequential([
           keras.Input(shape=(784,)),
           keras.layers.Reshape((28, 28, 1)),
           keras.layers.Conv2D(16, (3, 3), activation="relu"),
           keras.layers.Flatten(),
           keras.layers.Dense(10, activation="softmax"),
       ])
       model.compile(optimizer="adam", loss="categorical_crossentropy")
       return model
   ```

2. **Instantiate and Run**

   ```python
   from parallel_finder import ParallelFinder
   from tensorflow.keras.datasets import mnist
   from tensorflow.keras.utils import to_categorical

   # Load data
   (x_train, y_train), _ = mnist.load_data()
   x_train = x_train.reshape(-1, 784).astype("float32") / 255
   y_train = to_categorical(y_train, 10)

   # List of model-building functions
   model_constructors = [build_dense, build_cnn]

   # Create finder
   finder = ParallelFinder(model_constructors)

   # Train each model for 5 epochs, batch size 64
   finder.find(
       train_data=x_train,
       train_labels=y_train,
       epochs=5,
       batch_size=64
   )

   # Inspect results
   for key, val in finder.logs.items():
       print(f"{key}: {val}")
   ```

3. **Interpret `finder.logs`**

   * Per-model entries (for N models):

     * `model_0_loss`, `model_0_time`, ..., `model_{N-1}_loss`, `model_{N-1}_time`
   * Global best entries:

     * `best_loss` → smallest final loss
     * `best_loss_model_idx` → index of model with `best_loss`
     * `time_for_best_loss` → epoch time for that model
     * `best_time` → shortest final-epoch time
     * `best_time_model_idx` → index of model with `best_time`
     * `loss_for_best_time` → loss corresponding to `best_time`

---

## API Summary

```python
class ParallelFinder:
    def __init__(self, model_fn_list: List[Callable[[], keras.Model]]):
        # model_fn_list: a list of functions that return compiled Keras models.
        # Initializes shared logs and a lock for safe multiprocessing.

    def find(self,
             train_data,
             train_labels,
             epochs: int = 1,
             batch_size: int = 32,
             **kw_fit):
        # Launches one process per model in model_fn_list.
        # Each process runs model.fit(train_data, train_labels, epochs, batch_size, callbacks=[FinderCallback]).
        # Returns None; results are in self.logs.
```

* **Important**

  * Child processes share `finder.logs` via a `multiprocessing.Manager().dict()`.
  * `FinderCallback` timestamps each epoch and, on the last epoch, writes metrics under a lock.

---

## Notes

* **GPU Usage**: If multiple processes share a GPU, configure each model to allow memory growth (e.g., `tf.config.experimental.set_memory_growth`).
* **Data Overhead**: Large datasets are pickled for each process—consider lightweight datasets or shared memory solutions if this becomes a bottleneck.
* **Customization**:

  * Modify `FinderCallback.on_epoch_end` to track validation metrics (e.g., `'val_loss'`).
  * Pass additional `fit(...)` arguments via `**kw_fit` (e.g., `validation_data=(x_val, y_val)`).
