---
layout: post
title: Logging Parameters of Deep Learning Models with AI-JSONable
---

In the two previous posts ([here]({{ site.url }}/Packaging-Deep-Learning-Models-with-Pickle/) and [here]({{ site.url }}/Packaging-Deep-Learning-Models-with-Dill/)) I covered how to serialize end-2-end deep-learning models, including pre- and post-processing, using respectively `pickle` and `dill`. We saw that both methods are subject to the shortcoming that the saved output is opaque as regarding which parameters and settings were used to build a model. This is often vital information for collaboration, post-mortem debugging, etc..

In this post, we review a method for tracking parameters. It's based on a package, `aijson` I wrote myself in Python. The package is less than 350 lines of Python code, and demonstrates how to separate saving model output, from tracking parameters. Interested parties are open to either collaborate on the project, or write their own code based on this simple concept.

We install the package using `pip install ai-jsonable`.

The basic idea here is to track the inputs to function signatures and class `__init__` methods, and to log all jsonable items. If an item is not jsonable, we chase down its initialization functions or methods, and do the same there and so forth.

## NLP example continued

Let's continue with the same model we used last time. We'll modify the code slightly, to include a few more interesting parameters, and we'll add the `@aijson` decorator.

```python
import io
import dill
import pickle
import sentencepiece
import sys
import torch

from aijson import aijson


@aijson
class MyTokenizer:
    def __init__(self, calibration_file, vocab_size=100):
        self.model = io.BytesIO()
        self.vocab_size = vocab_size
        self.calibrate(calibration_file)
        self.n_pieces = self.tokenizer.get_piece_size()

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

      
@aijson
class MyReader:
    def __init__(self, d):
        self.d = d

    def __call__(self, x):
        pred = x.topk(1)[1].item()
        return {'decision': self.d[pred], 'prediction': x.tolist()}


@aijson
class MyLayer(torch.nn.Module):
    def __init__(self, n_symbols, n_classes, n_embed=64, n_hidden=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, n_embed)
        self.rnn = torch.nn.GRU(n_embed, n_hidden, 1, batch_first=True,
                                dropout=0)
        self.proj = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        return self.proj(self.rnn(self.embedding(x))[0])


@aijson
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

Now let's set up the model, including logging our parameters:

```python
from aijson import logging_context
import json


with logging_context as lc:
    i = MyTokenizer()
    l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
    out = MyReader(['good', 'bad', 'ugly'])
    c = MyCompoundClass(i, l_, out)
    

print(json.dumps(lc, indent=2))
```

You'll see that the JSON output gives a recursive definition of the full model:

```json
{
  "var0": {
    "module": "mymodels",
    "caller": "MyTokenizer",
    "kwargs": {
      "calibration_file": "corpus.txt",
      "vocab_size": 100
    }
  },
  "var1": {
    "module": "mymodels",
    "caller": "MyLayer",
    "kwargs": {
      "n_symbols": 100,
      "n_classes": 3
    }
  },
  "var2": {
    "module": "mymodels",
    "caller": "MyReader",
    "kwargs": {
      "d": [
        "good",
        "bad",
        "ugly"
      ]
    }
  },
  "var3": {
    "module": "mymodels",
    "caller": "MyCompoundClass",
    "kwargs": {
      "tokenizer": "$var0",
      "layer": "$var1",
      "decoder": "$var2"
    }
  }
}
```

If we want to reinstantiate the model based on the JSONable output, we can do that without recourse to the original Python build code. If needs be, parameters can be changed in-line by modifying `lc` directly.

```python
from aijson.build import build

c = build(lc)
```

This method can be combined with the merits of `pickle` or `dill` to get a nice experiment tracking, logging and serialization system. All parameters which go into building a model (including training code) can be logged with `aijson`. Inside the training loop, we can use `pickle` or `dill` to serialize the model to a data blob. If any doubts about parameter values arise, or these need to be accessed programmatically, these can be gleaned directly from the JSON output.

## Modular experimentation, including pre- and post-processing

Using `aijson` together with writing configuration files in Python, which may be used to flexibly "wire" together models, we get a very flexible experimentation environment. There need be no pre-conceived notion of which pre-processor goes together with which forward pass and which post-processor. This logic may be carried over further to the inner mechanics of each of these components. When exporting the results of the experiment, the parameters are logged with `aijson` without needing to add cumbersome building or logging boilerplate to the code.

## Next time...

Next week I'll be looking at a different approach which replaces `pickle`, `dill` and `aijson`, and allows for very transparent model serialization and parameterisation. Stay tuned...