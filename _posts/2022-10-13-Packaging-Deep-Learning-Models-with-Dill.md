---
layout: post
title: Packaging Deep Learning Models with Dill
---

A  [few posts back]({{ site.url }}/Packaging-Deep-Learning-Models-is-Hard/) I announced a series of posts examining how to serialize deep-learning models with a variety of techniques. [In the previous post]({{ site.url }}/Packaging-Deep-Learning-Models-with-Pickle/), we looked at the standard method for serializing Python objects, using `pickle`, which is in the Python standard library. We saw that `pickle` is subject to several gotchas and drawbacks when it comes to deep learning code. In this post we're going to look at `dill`, a third party package, developed by the [The UQ Foundation](https://github.com/uqfoundation). `dill` purports to solve several issues which arise when using `pickle`. We'll see in the post, that how to actually use `dill` to do this is not exactly straightforward, and comes with some additional gotchas, which aren't necessarily intuitive on the outset. As in the previous post I'm going to use these simple classes to instantiate our toy-NLP model:

In `package.py`:

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

As before we can set up the model like this:

```python
i = MyTokenizer()
i.calibrate('corpus.txt')
l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
out = MyReader(['good', 'bad', 'ugly'])
c = MyCompoundClass(i, l_, out)
```

We saw that with `pickle` a major drawback is that the *source* code is not serialized together with the object data. In many applications, we may want a stable and secure store of models generated in past experiments, including the exact methods and routines which went into those experiments. Often times, source code may change between experiments, in order to cater to the insights developed in previous experiments. `dill` may be used to get around this drawback, by enabling objects, including source code, to be dumped along with the object data. How exactly to use `dill` to do this, however, is not exactly straightforward, and I found the [documentation]() not exactly enlightening on this topic. The [GitHub page](https://github.com/uqfoundation/dill) mentions front and centrally, that `dill` provides the capability to "save and extract the source code from functions and classes". However there seem to be no clear examples of usage in the documentation or GitHub for this use case. For this reason it took me a bit of playing around to apply `dill` for this case.

Here is how it's possible to get `dill` to save the source code of a model, together with the data. Further down in `package.py`, we write this:

```python
if __name__ == '__main__':
    print(apply_model(c, 'lorem ipsum'))

    with open('model.dill', 'wb') as f:
    		dill.dump(c, f)
```

We get this output:

```
{'decision': 'good', 'prediction': [1.9316778182983398, 1.865881085395813, 1.9164676666259766]}
```

Now, in a new program, we can reload the model as follows. In `load.py`:

```python
import dill

from torchapply import apply_model

with open('model.dill', 'rb') as f:
    c = dill.load(f, ignore=True)

print(apply_model(c, 'lorem ipsum'))
```

We get the same output, just as we wanted:

```
{'decision': 'good', 'prediction': [1.9316778182983398, 1.865881085395813, 1.9164676666259766]}
```

Now let's modify the source code, changing the `forward` method of the `MyCompoundClass` object:

```python
def forward(self, args):
    output = self.layer(args)[:, -1, :]
    print('testing testing 123')
    return self.function(output)
```

We are hoping to have saved the source, code. Let's try reloading the model, as per the docs, like this:

```python
import dill

from torchapply import apply_model

with open('model.dill', 'rb') as f:
    c = dill.load(f)

print(apply_model(c, 'lorem ipsum'))
```

Hoping to reload the model as in `pickle`, we instead get this error:

```
Traceback (most recent call last):
  File "/Users/dodo/blythed/blythed.github.io/code/2022-10-13-Packaging-Deep-Learning-Models-with-Dill/load.py", line 6, in <module>
    c = dill.load(f)
  File "/Users/dodo/blythed/superduperdb/.venv/lib/python3.10/site-packages/dill/_dill.py", line 373, in load
    return Unpickler(file, ignore=ignore, **kwds).load()
  File "/Users/dodo/blythed/superduperdb/.venv/lib/python3.10/site-packages/dill/_dill.py", line 646, in load
    obj = StockUnpickler.load(self)
  File "/Users/dodo/blythed/superduperdb/.venv/lib/python3.10/site-packages/dill/_dill.py", line 636, in find_class
    return StockUnpickler.find_class(self, module, name)
AttributeError: Can't get attribute 'MyCompoundClass' on <module '__main__' from '/Users/dodo/blythed/blythed.github.io/code/2022-10-13-Packaging-Deep-Learning-Models-with-Dill/load.py'>
```

In order to make this work, we need to change the serialization code in this way:

```python
with open('model.dill', 'wb') as f:
    dill.dump(c, f, byref=False)
```

After doing this, we get this output from `python save.py`:

```
{'decision': 'good', 'prediction': [1.9316778182983398, 1.865881085395813, 1.9164676666259766]}
```

This shows that the new rewritten `forward` method was not applied; this is exactly what we wanted, to be able to use the *old code* rather than the new code.
Now things start getting a little tricky. Suppose that instead of serializing the model in the `if __name__ == '__main__'` construct, I had done this instead in a new program `save.py`, then we would get different behaviour. So in `save.py` we would have:

```python
from package import *
from torchapply import apply_model
import dill


i = MyTokenizer()
i.calibrate('corpus.txt')
l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
out = MyReader(['good', 'bad', 'ugly'])
c = MyCompoundClass(i, l_, out)
c.eval()

print(apply_model(c, 'lorem ipsum'))
with open('model.dill', 'wb') as f:
    dill.dump(c, f, byref=False)

```

Then a call to `python load.py` after modifying the source code as above would give:

```
testing testing 123
{'decision': 'good', 'prediction': [1.9316778182983398, 1.865881085395813, 1.9164676666259766]}
```

So clearly, in this case, the `dill` serialization routine, *does not* save the object code as well as data. It seems this has to do with whether an object has been defined in the same namespace as the code serializing an object, or rather has been imported from another package or module. This is a rather unexpected behaviour, and often not identical to the type of persistence I'm looking for in deep learning experimentation. Often, I may want to serialize *all* functions and classes within a certain scope, but not restrict myself in such a rigid way, so that the only acceptable scope is `__main__`. I see that there are discussions and have been features proposed on the `dill` forums with regards to [adding the ability to go deeper](https://github.com/uqfoundation/dill/pull/47), and recursively serialize the object in question. However, these proposals were slated due to performance concerns. I personally would still like to make that decision for myself, but, *hey -- not my package!*

Another drawback shared by `pickle` is that we still don't know, even if we are successfully able to save object code, what the key parameters and setting are in the code which went into defining the object. The `dill` and `pickle` objects simply exist as a blob of data, ready to be shared and sent over a network etc.. Historically `dill` and `pickle` were not intended as model serialization tools, *per se*. So no special consideration of these types of delicacies were considered. Notwithstanding this, both PyTorch and Scikit-Learn, two of the most widely used machine learning packages in Python, both make recourse to `pickle`.

In the following two posts, I am going to cover 2 tools which address the problems discussed here, namely, code persistence and parameter transparency. **Stay tuned!**

