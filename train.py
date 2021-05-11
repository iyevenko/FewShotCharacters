import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Dense, Concatenate, Flatten


class AutoCompileModel():
    def _compile(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


class LayeredModel(tf.keras.models.Model, AutoCompileModel):

    def __init__(self, layer_widths, num_classes=10):
        super().__init__()
        self._compile()
        self.flatten = Flatten()
        self.dense_layers = [
            Dense(w, activation='relu') for w in layer_widths
        ]
        self.out = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        i = 0
        activations = tf.TensorArray(dtype=tf.float32, size=len(self.dense_layers))

        X = self.flatten(inputs)

        for ffn in self.dense_layers:
            X = ffn(X)
            activations.write(i, X)
            i+=1

        X = self.out(X)

        return X, activations

class BackwardConnectedModel(tf.keras.models.Model, AutoCompileModel):

    def __init__(self, layer_widths, num_classes=10):
        super().__init__()
        self._compile()
        self.flatten = Flatten()
        self.dense_layers = [
            Dense(w, activation='relu') for w in layer_widths
        ]
        self.out = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # Cache activations from most recent pass through the model
        i = 0
        activations = tf.TensorArray(dtype=tf.float32, size=len(self.dense_layers))

        X = self.flatten(inputs)

        for ffn in self.dense_layers:
            X = tf.concat([X, ffn(X)], axis=-1)
            activations.write(i, X)
            i+=1

        X = self.out(X)

        return X, activations

    def prune_weights(self, percentile, inclue_biases=False):
        pass


class TopFeatureClassifier(tf.keras.models.Model, AutoCompileModel):

    def __init__(self, base_model):
        super().__init__()
        self._compile()
        self.base_model = base_model
        self.dense = Dense(1)

    def call(self, inputs, training=None, mask=None):
        X, activations = self.base_model(inputs)
        Y = self.dense(activations.read(activations.size()-1))
        X = tf.concat([X, Y], axis=-1)
        # activations.close()

        return X


class FullFeatureClassifier(tf.keras.models.Model, AutoCompileModel):

    def __init__(self, base_model):
        super().__init__()
        self._compile()
        self.base_model = base_model
        self.dense = Dense(1)

    def call(self, inputs, training=None, mask=None):
        X = self.base_model(inputs)
        Y = self.dense(tf.stack(self.base_model.activations.concat()))
        X = tf.concat([X, Y], axis=-1)

        return X


def load_ds():
    ds_train, ds_test=  tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    ds_train = ds_train.map(lambda x, y: (tf.cast(x, tf.float32)/255., y))

    full_train = ds_train.batch(1)
    no_holdout_train = ds_train.filter(lambda x, y: tf.not_equal(y, 9)).batch(1)
    holdout_train = ds_train.filter(lambda x, y: tf.equal(y, 9)).batch(1)

    test_nums = []
    for i in range(10):
        test_nums.append(ds_test.filter(lambda x, y: tf.equal(y, i)).batch(1))

    return full_train, no_holdout_train, holdout_train, test_nums


def train_model(model, ds, steps=None, start=0):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            out = model(x)
            if isinstance(out, tuple):
                out, activations = out
                loss = model.loss_fn(y, out)
                # activations.close()
            else:
                loss = model.loss_fn(y, out)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        model.accuracy.update_state(y, out)
        return loss

    i = 0
    history = []
    ws = []
    for step, (x, y) in enumerate(ds.skip(start)):
        loss = train_step(x, y)
        history.append(loss)

        if hasattr(model, 'dense'):
            ws.append(model.dense.weights[0].numpy())

        if steps != None and i >= steps:
            break
        i += 1

    return history, np.stack(ws) if ws != [] else None


def eval_on_nums(model, nums):
    @tf.function
    def test_step(x, y):
        out = model(x, training=False)
        if isinstance(out, tuple):
            out, activations = out
            loss = model.loss_fn(y, out)
            # activations.close()
        else:
            loss = model.loss_fn(y, out)

        model.accuracy.update_state(y, out)
        return loss


    accuracy = []
    for ds in nums:
        for step, (x, y) in enumerate(ds):
            test_step(x, y)

        accuracy.append(float(model.accuracy.result().numpy()))
        model.accuracy.reset_states()


    return accuracy


if __name__ == '__main__':
    tf.random.set_seed(0)

    full_train, no_holdout_train, holdout_train, test_nums = load_ds()

    layer_widths = [64, 64, 64]
    #
    # # ------------------------------ Layered ------------------------------ #
    holdout_steps = 10000
    heldout_model = LayeredModel(layer_widths, num_classes=9)
    train_model(heldout_model, no_holdout_train, steps=holdout_steps)
    heldout_model.trainable = False



    # Baseline
    # baseline_model = LayeredModel(layer_widths)
    # history = train_model(baseline_model, full_train, steps=1000)
    # accuracy = eval_on_nums(baseline_model, test_nums)
    # print(accuracy)

    # Highest level features
    ws = []
    steps  = 1000
    top_classifier = TopFeatureClassifier(heldout_model)
    history, wt = train_model(top_classifier, holdout_train, steps=steps, start=0)

    plt.plot(history)
    plt.show()
    plt.close()

    plt.imshow(np.transpose(wt, [1, 0, 2]))
    plt.show()
    plt.close()
    # for i in range(20):
    #     train_model(top_classifier, full_train, steps=steps, start=steps*i + holdout_steps)
    #     ws.append(top_classifier.dense.weights[0].numpy())
    #     # accuracy = eval_on_nums(top_classifier, test_nums)
    #     # plt.title(f'Steps={steps}')
    #     # plt.bar(np.arange(10), accuracy)
    #     # plt.show()
    #     # plt.close()
    # #
    # wt = np.stack(ws)
    # plt.imshow(np.transpose(wt, [1, 0, 2]))
    # plt.show()


    # All level features
    # steps = 10
    # top_classifier = FullFeatureClassifier(heldout_model)
    # compile_model(top_classifier)
    # for i in range(5):
    #     train_model(top_classifier, full_train, steps=steps, start=steps * i)
    #     accuracy = eval_on_nums(top_classifier, test_nums)
    #     plt.title(f'Steps={steps}')
    #     plt.bar(np.arange(10), accuracy)
    #     plt.show()
