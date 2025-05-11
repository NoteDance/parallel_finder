# ParallelFinder

## Overview

The `ModelFinder` utility lets you train multiple Keras models in parallel (each in its own process), monitor their per‑epoch losses via a shared callback, and automatically record the optimizer configuration that achieved the lowest final loss. You supply a list of pre‑compiled `tf.keras` models and a training dataset; `ModelFinder.find(...)` will spin up one process per model, run `model.fit(...)` with a custom `LambdaCallback` that captures losses in a `multiprocessing.Manager().dict()`, and after all processes join you can inspect `finder.logs['best_loss']` and `finder.logs['best_opt']` to see which optimizer and loss prevailed.

## API Reference

### `epoch_end_callback(epoch, logs, model, lock, callback_func)`

A thin wrapper that unpacks arguments for the real callback. Internally it simply calls:

```python
callback_func(epoch, logs, model, lock)
```

Use this to bind your model, lock, and callback function into a signature that `LambdaCallback(on_epoch_end=…)` expects.

### Class `ModelFinder`

| Member                                                         | Description                                                                                                                                                                                                          |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__(self, models)`                                       | Store a list of compiled Keras models. Create a `multiprocessing.Manager().dict()` named `logs`, initialize `logs['best_loss']=1e9`, and create a `Lock` for safe updates.                                           |
| `logs`                                                         | A process‑shared dictionary. After training, contains `best_loss` (float) and `best_opt` (optimizer).                                                                                                                |
| `lock`                                                         | A `multiprocessing.Lock` used to serialize access to `logs`.                                                                                                                                                         |
| `on_epoch_end(self, epoch, logs, model=None, lock=None)`       | Callback invoked at the end of each epoch. If it’s the final epoch (`epoch+1 == self.epochs`) and the reported `logs['loss']` is lower than the current `logs['best_loss']`, update both `best_loss` and `best_opt`. |
| `find(self, train_ds=None, batch_size=64, epochs=1, **kw_fit)` | The main entry point. Sets `self.epochs`, then for each model:                                                                                                                                                       |

> 1. Create a `partial(epoch_end_callback, model=…, lock=…, callback_func=self.on_epoch_end)`.
> 2. Wrap that in `LambdaCallback(on_epoch_end=…)`.
> 3. Launch `multiprocessing.Process(target=model.fit, kwargs={…,'callbacks':[that callback],…})`.
> 4. Start and collect all processes, then `join()` them.
>    After all processes complete, `self.logs` holds the best optimizer and loss. |

## Example Usage

```python
from keras.models import Sequential
from keras.layers import Dense
from parallel_finder import ParallelFinder

# 1) Build some models with different learning rates
models = []
for lr in [1e-2, 1e-3, 1e-4]:
    m = Sequential([Dense(16, activation='relu'), Dense(1)])
    m.compile(optimizer='adam', loss='mse', learning_rate=lr)
    models.append(m)

# 2) Instantiate ModelFinder
finder = ParallelFinder(models)

# 3) Run parallel training for 5 epochs on (x_train, y_train)
finder.find(train_ds=(x_train, y_train), batch_size=32, epochs=5)

# 4) Inspect results
print("Best loss:", finder.logs['best_loss'])
print("Best optimizer config:", finder.logs['best_loss_model'].optimizer.get_config())
```
```python
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam
from parallel_finder import ParallelFinder

# 1) Define base model architecture
def build_base_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(1)
    ])
    return model

# 2) Instantiate models with different optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adam': Adam(learning_rate=0.001),
}

models = []
for name, opt in optimizers.items():
    m = build_base_model()
    # compile each clone with its optimizer
    m.compile(optimizer=opt, loss='mse')
    models.append(m)

# 3) Run ModelFinder on (x_train, y_train)
finder = ParallelFinder(models)
finder.find(train_ds=(x_train, y_train), batch_size=32, epochs=10)

# 4) Report results
print("Best loss:", finder.logs['best_loss'])
print("Best optimizer:", finder.logs['best_loss_model'].optimizer.get_config()['name'])
```

This will print the smallest final loss achieved and the full configuration of the optimizer that achieved it.&#x20;

## Key Concepts and Best Practices

### Shared State with `multiprocessing.Manager().dict()`

* A manager‑backed dict lives in a server process; worker processes hold proxies that forward get/set operations safely. This avoids race conditions when multiple processes update a Python dict concurrently.&#x20;

### Serializing Keras Models for Multiprocessing

* On Windows (spawn start method) your models, optimizer state, and data must be pickle‑able. If you run into pickling errors, consider moving model construction inside the spawned process or switch to a Linux/macOS fork‑based start.

### GPU Considerations

* Multiple processes sharing one GPU can contend for VRAM. In multi‑GPU environments, set the `CUDA_VISIBLE_DEVICES` environment variable per process, or use TensorFlow’s per‑process GPU memory growth settings to avoid out‑of‑memory errors.

### Extending the Callback

* You can easily record additional metrics (e.g. `logs['val_loss']`, `logs['accuracy']`) by reading them in `on_epoch_end` and updating `self.logs` with your own selection logic.
