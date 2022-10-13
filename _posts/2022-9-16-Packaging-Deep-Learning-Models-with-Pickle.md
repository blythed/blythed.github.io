---
layout: post
title: Packaging Deep Learning Models with Pickle
---

We saw [two posts back]({{ site.url }}/Packaging-Deep-Learning-Models-is-Hard/) that it's important when productionizing, and crystallizing the results of deep-learning experimentation that all logic necessary for production is encapsulated in a single serialized format. If that's not achieved, then data engineers may be confronted with awkward questions such as:

- How do I prepare inputs and postprocess outputs for the forward pass?
- How should decoding be performed for this model?
- What parameters were used to instantiate this model?
- What code was used to produce this model?

In this post, let's talk about using the `pickle` module for serializing a model, including pre- and post-processing. First let's introduce a toy NLP model, which 
has some non-trivial elements. We have:

- Data (the tokenizer state) not handled by the `torch` serializer
- A model design pattern including pre- and post-processing

```python
import io
import dill
import pickle
import sentencepiece
import sys
import torch


class MyTokenizer:
    def __init__(self, vocab_size=100):
        self.model = io.BytesIO()
        self.vocab_size = vocab_size

    def calibrate(self, file):
        with open(file, 'r') as f:
            sentencepiece.SentencePieceTrainer.train(
                sentence_iterator=iter(f.readlines()),
                model_writer=self.model,
                vocab_size=self.vocab_size,
            )
        self.tokenizer =  sentencepiece.SentencePieceProcessor(model_proto=self.model.getvalue())

    def __call__(self, x):
        return self.tokenizer.encode(x)


class MyReader:
    def __init__(self, d):
        self.d = d

    def __call__(self, x):
        pred = x.topk(1)[1].item()
        return {'decision': self.d[pred], 'prediction': x.tolist()}


class MyLayer(torch.nn.Module):
    def __init__(self, n_symbols, n_classes, n_embed=64, n_hidden=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, n_embed)
        self.rnn = torch.nn.GRU(n_embed, n_hidden, 1, batch_first=True,
                                dropout=0)
        self.proj = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        return self.proj(self.rnn(self.embedding(x))[0])


class MyCompoundClass(torch.nn.Module):
    def __init__(self, tokenizer, layer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder

    def preprocess(self, arg):
        return torch.tensor(self.tokenizer(arg))

    def forward(self, args):
        output = self.layer(args)[:, -1, :]
        return output

    def postprocess(self, x):
        return self.decoder(x)
```

We can set this model up like this, calibrating the tokenizer and instantiating the other layers:

```python
>>> i = MyTokenizer()
>>> i.calibrate('corpus.txt')
>>> l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
>>> out = MyReader(['good', 'bad', 'ugly'])
>>> c = MyCompoundClass(i, l_, out)
```

This model can be applied to single or multiple data points using the design pattern we described in the [previous post]({{ site.url }}/Wrap-Your-PyTorch-Models-With-This-Function-For-Great-Convenience/).

```python
>>> print(apply_model(c, 'lorem ipsum'))
>>> print(apply_model(c, ['lorem ipsum', 'lorem ipsum'], single=False, batch_size=2))
```

Using the `apply_model` design pattern is expedient when our model is not in some sense reducible to simply the model weights and hyperparameters; the pattern allows us to incorporate all additional data and subroutines for applying preprocessing and postprocessing.

Now that the model is set up, we can save the model using the `pickle` module:

```python
>>> with open('model.pkl', 'wb') as f:
...     pickle.dump(c, f)
```

If we reload the model with `pickle` then we recover the entire object `c` including the necessary auxiliary data associated with it necessary for, e.g., the tokenizer:

```python
>>> open('model.pkl', 'rb') as f:
...     reload_c = pickle.load(f)
```

We can affirm that this save/ load was successful:

```python
>>> assert apply_model(c, 'lorem ipsum') == apply_model(reload_c, 'lorem ipsum')
```

## Caveats

There are a few caveats to be aware of when applying `pickle`.

### Code changes

Suppose, for instance, that we change the indexing in the `forward` method of `MyCompoundClass`:

```python
def forward(self, args):
		output = self.layer(args)[:, -20, :]
  	return output
```

Now let's try reloading the model and testing it as above. We get this error message:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/package.py", line 88, in <module>
    print(apply_model(c, 'lorem ipsum'))
  File "/Users/dodo/blythed/superduperdb/.venv/lib/python3.10/site-packages/torchapply/apply.py", line 95, in apply_model
    output = model.forward(singleton_batch)
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/package.py", line 69, in forward
    output = self.layer(args)[:, -20, :]
IndexError: index -20 is out of bounds for dimension 1 with size 7
```

Without having changed the saved `.pkl` file, we see that the modified code is being applied.

### Renaming objects

Similarly, if we rename the package or module in which we defined our functions and classes, then we get this error:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/load.py", line 9, in <module>
    c = pickle.load(f)
ModuleNotFoundError: No module named 'package'
```

In addition, if the object `c` is built in the same module as the objects are defined (using the `if __name__ == "__main__"` construct) we get a similar error:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/load.py", line 9, in <module>
    c = pickle.load(f)
AttributeError: Can't get attribute 'MyCompoundClass' on <module '__main__' from '/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/load.py'>
```

### Unsupported python objects

What objects does `pickle` support? `pickle` supports a range of classes and functions, however there are several types of objects `pickle` does not support. These include classes and functions including locally defined functions, including `lambda` functions.

For example if we add an additional attribute, implementing a `lambda` to the definition of `MyCompoundClass` then we get an error:

```python
class MyCompoundClass(torch.nn.Module):
    def __init__(self, tokenizer, layer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder
        self.function = lambda x: x
```

 When we try and save the model as above we get this error:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/package.py", line 104, in <module>
    pickle.dump(c, f)
AttributeError: Can't pickle local object 'MyCompoundClass.__init__.<locals>.<lambda>'
```

There is a recursive definition of what is pickleable and what is not in the `pickle` [documentation](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled). The `lambda` gotcha is one that I've stumbled up and against time and again, as I like to use local functions.

### Custom pickling

The standard way to fix such unpickleable attributes, is to overwrite the `__getstate__` and `__setstate__` methods:

```python
class MyCompoundClass(torch.nn.Module):
    def __init__(self, tokenizer, layer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder
        self.function = lambda x: x

    def __getstate__(self):
        return {'t': self.tokenizer, 'l': self.layer, 'd': self.decoder}

    def __setstate__(self, state):
        self.tokenizer = state['t']
        self.layer = state['l']
        self.decoder = state['d']
```

Officially by writing the class like this, the attribute `self.function` is ignored by `pickle` when saving and loading and so is not a problem. However, when I tried this with the above example I get this issue:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/package.py", line 85, in <module>
    c = pickle.load(f)
  File "/Users/dodo/blythed/blythed.github.io/code/2022-9-16-Packaging-Deep-Learning-Models-with-Pickle/package.py", line 66, in __setstate__
    self.layer = state['l']
  File "/Users/dodo/blythed/superduperdb/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1236, in __setattr__
    raise AttributeError(
AttributeError: cannot assign module before Module.__init__() call
```

The workaround I settled for in this example, was to **not** subclass `torch.nn.Module` for the main `MyCompoundClass`:

```python
class MyCompoundClass:
    def __init__(self, tokenizer, layer, decoder):
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder
        self.function = lambda x: x

    def eval(self):
        self.layer.eval()

    def train(self):
        self.layer.train()

    def __getstate__(self):
        return {
            't': self.tokenizer,
            'l': self.layer,
            'd': self.decoder,
        }

    def __setstate__(self, state):
        self.tokenizer = state['t']
        self.layer = state['l']
        self.decoder = state['d']

    ...
```

I'm not totally happy with this solution, but it gets the job done. The `eval` and `train` methods are necessary for `apply_model`.

## Verdict

What's great about this approach?

- `pickle` is a Python standasrd library native approach to object serialization, assuring continued support, predictability and reliability.

- Many scientific libraries have based their serialization routines on `pickle`, for instance Scikit-Learn and PyTorch. Using `pickle` means we'll be in good company.

There are several issues with security for `pickle` which are well known. I won't discuss those here. What are the other disadvantages of this approach, specific to model building and serialization?

- `pickle` saves the  structure recursively which is necessary for reconstructing the saved object. The saved output, however, does not allow one to view this structure which is in any way transparent. In many use cases, it's very useful to have an explicit view of the parameters and settings which went into building a model, as this allows debugging and experimentation to proceed based on those values.
- If the data scientist would like to persist not just the data associated with the model, but also the code, then the `pickle` method won't suffice. That's because `pickle` imports the functions and classes referred to in the `pickle` saved output. It does not save versions of those functions and classes explicitly. For many use cases, this is not the desired behaviour. Saving models is often about pinning *everything* which went into the model development. This is in contrast to the third party package `dill` which can be applied to save function and class definitions also.

In the next blog post I'll talk about using `dill` for the same serialization task, avoiding this second shortcoming.