# Mixed History Recurrent Neural Networks

This is the accompanying code for

DiPietro, Robert, Nassir Navab, and Gregory D. Hager. "Revisiting NARX
Recurrent Neural Networks for Long-Term Dependencies." arXiv preprint
arXiv:1702.07805 (2017). https://arxiv.org/abs/1702.07805

We ask that you cite the paper if you find the code useful in your research.

### High-Level Overview of the Paper

- RNNs struggle with learning long-term dependencies because of the vanishing
gradient problem, which can't be resolved in the absolute (see Bengio et al.,
1994).

- LSTM and GRUs use one specific mechanism to help alleviate this problem. NARX
RNNs take an entirely different approach, by including direct connections to
the past.

- We analyze the vanishing gradient problem for NARX RNNs in detail, and based
on this analysis introduce a new variant which we call MIxed hiSTory RNNs (MIST
RNNs).

- We compare simple RNNs, LSTM, GRUs, and MIST RNNs across 4 diverse tasks. MIST
RNNs significantly outperform LSTM and GRUs in 2 cases and match performance in
the other 2 cases.

### Overview of this Repository

#### Models

See `layers.py` and `models.py`. Here you'll find
from-scratch implementations of simple RNNs, LSTM, GRUs, and MIST RNNs.

(Note that our LSTM implementation matches results in prior work, and our GRU
implementation improves these results further.)

Why from scratch?

MIST RNNs don't fit the typical `RNNCell`, `dynamic_rnn` approach because they
depend on many states from the past. Though this is only true for MIST RNNs, we
prefer unified code that handles all experiments.

Also, for research we favor concise, somewhat special-purpose code over bulky,
general-purpose code. For example, [models.py](models.py) is about 100 lines of
code without comments, and it handles classification, regression, and
multi-label classification RNNs, in all cases handling both sequence to sequence
mappings and sequence to value mappings. Compare this to TensorFlow Learn's
[dynamic_rnn_estimator.py](https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/learn/python/learn/estimators/dynamic_rnn_estimator.py)
or Keras's
[recurrent.py](https://github.com/fchollet/keras/blob/tf-keras/keras/layers/recurrent.py).

That said, we do hope to provide an `RNNCell` in the future; it should be able
to work with TensorFlow's `raw_rnn`, which right now is in an early testing
phase with an API that's not yet stable.

#### Data: Downloading, Generating, Preprocessing

See `copyproblem.py`, `additionproblem.py`, `timit.py`, `timitphonemerec.py`,
and `mnist.py`.

All of these files are executables (for downloading / generating /
preprocessing), and all except `timit.py` are also modules which make it
easy to load train, val, test splits.

Example:

```
(python3.5)rdipiet2@thin6 mist-rnns $ python3 mnist.py -h
usage: mnist.py [-h] [--data_dir DATA_DIR]

Download MNIST.

optional arguments:
  ...
```

All arguments have defaults and are therefore optional. Here, `data_dir`
defaults to `~/Data/MNIST`.

After running `mnist.py` to download the data, we can load it with (for
example)

```
import mnist
outs = mnist.load_split(val=True, permute=True, normalize=True, num_val=2000)
train_images, train_labels, val_images, val_labels = outs
```

`copyproblem.py`, `additionproblem.py`, `mnist.py` give immediate access to data
for the copy problem, addition problem, and MNIST tasks.

Unfortunately we can't give immediate access to TIMIT because it's not freely
available. Instead, `timit.py` processes the `NIST Speech Disc CD1-1.1` release
to form the standard train, val, test sets (see paper or code for details).
`timitphonemerec.py` then processes this data further, producing MFCC
coefficients etc. Finally `timitphonemerec` provides the same `load_split`
functionality as elsewhere.

#### Training

See `copyproblem_train.py`, `additionproblem_train.py`,
`timitphonemerec_train.py`, and `mnist_train.py`. Each is an executable for
training and exporting summaries / results on the train and val sets.

Example:

```
(python3.5)rdipiet2@thin6 mist-rnns $ python3 mnist_train.py -h
usage: mnist_train.py [-h] [--data_dir DATA_DIR] [--debug DEBUG]
                      [--permute PERMUTE] [--layer_type LAYER_TYPE]
                      [--activation_type ACTIVATION_TYPE]
                      [--num_hidden_units NUM_HIDDEN_UNITS]
                      [--optimizer OPTIMIZER] [--learning_rate LEARNING_RATE]
                      [--optional_bias_shift OPTIONAL_BIAS_SHIFT]
                      [--num_pre_act_mixture_delays NUM_PRE_ACT_MIXTURE_DELAYS]
                      [--trial TRIAL]

Train an RNN for sequential (possibly permuted) MNIST recognition.

optional arguments:
  ...
```

Again, all arguments have defaults and are therefore optional. `layer_type` can
be any layer from `layers.py`: `SimpleLayer`, `LSTMLayer`, `GRULayer`, or
`MISTLayer`. Similarly, `optimizer` can be any optimizer from `optimizers.py`,
which right now includes `ClippingGradientDescentOptimizer` and
`ClippingMomentumOptimizer`, where in both cases 'clipping' really refers to
thresholded scaling (see Pascanu et al., 2013)

All TensorBoard summaries / models / other status files are written to
`DATA_DIR/results/dir_with_params_in_the_name`. Each run is then contained in
its own directory and can be analyzed together by navigating to
`DATA_DIR/results` with TensorBoard.

Important note: Do not run training with the default learning rate and expect to
see reasonable results. All models are sensitive to learning rate; see the paper
for good choices for various task, method pairs.

#### Batch Runs

See `batch_runs/`, which includes `copyproblem_template.sh`,
`additionproblem_template.sh`, `timitphonemerec_template.sh`, and
`mnist_template.sh`. By default, each run uses a learning rate sampled randomly
in log space over `10^-4` and `10^1`.

#### Testing

See `timitphonemerec_test.py` and `mnist_test.py`.

Example:

```
(python3.5)rdipiet2@thin6 mist-rnns $ python3 mnist_test.py -h
usage: mnist_test.py [-h] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR]

Test an RNN for sequential (possibly permuted) MNIST recognition.

optional arguments:
  ...
```

Here, `results_dir` is *required*, and must must be one of the directories
created during training, which contains TensorBoard summaries, a saved model,
etc.

#### Utils

See `utils.py`. This contains various helper functions to traverse examples
epoch by epoch, to pad sequences, to form batches for full BPTT or truncated
BPTT, etc.

#### Additional Notes

You may see slow performance if you use CPUs rather than GPUs. In particular,
TensorFlow for some reason leads to ~40% CPU utilization, and a trace shows that
the bottleneck is element-wise multiplication. If you resolve this issue with
compilation options, or if you implement MIST RNNs in Theano, Torch, etc.,
please let us know.

We used Nvidia K80s for all experiments.

#### Dependencies

Python 3 is required.

This code was upgraded to be compatible with the official TensorFlow 1.0
release, and was tested with this same version. It can be obtained via `pip
install tensorflow-gpu==1.0.0`.

Also, if you want to extract MFCC coefficients etc. for TIMIT, you'll need
`python-speech-features`. We used version 0.4, which you can get via
`pip install python-speech-features==0.4`.
