---
layout: post
title: Wrap your PyTorch models with this simple function for great convenience
---

In the [previous post]({{ site.url }}/Conventions-For-Packaging-Deep-Learning-Models/), I proposed a convention for packaging deep learning models, so that models always live together with the vital routines necessary for preparing and post-processing inputs and outputs for operational use. [Two posts back]({{ site.url }}/Packaging-Deep-Learning-Models-is-Hard/), I talked about why this is so important. In this post, I want to formalize this convention with a simple function written in pure PyTorch, which can really facilitate using PyTorch models in practice.

Before we do that, it's instructive to look at how PyTorch deals with complex and nested input in the `torch.utils.data.DataLoader` class:

```python
import torch.utils.data
import torch

datapoints = [
  [{'a': {'b': 1}, 'c': 2}, [0, 0]] for _ in range(10)
]

dataloader = torch.utils.data.DataLoader(datapoints, batch_size=2)

for batch in dataloader:
    print(batch)
```

We get this output:

```
[{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
[{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
[{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
[{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
[{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
```

You can see the `DataLoader` class drilling down into the "leaf" nodes of the individual data points, and batching at the level of those leaves. This works, provided the tree structure of the input is the same for all data points in the batch. This is rather convenient, since the nested structure of the datapoints may facilitate modular models, which operate at different levels of this tree structure.

For example we might have a model with preprocessing, forward pass and postprocessing, which looks something like this:

```python
import torch
from torch import tensor


class Main(torch.nn.Module):
    def __init__(self, model_0, model_1):
        super().__init__()
        self.model_0 = model_0
        self.model_1 = model_1
        self.dictionary = {'apple': 0, 'orange': 1, 'pear': 2}

    def preprocess(self, arg):
        return [
            {
                'a': {'b': self.dictionary[arg[0]['a']['b']]},
                'c': self.dictionary[arg[0]['c']]
            },
            torch.tensor([self.dictionary[x] for x in arg[1]])
        ]

    def forward(self, args):
        return self.model_0(args[0]), self.model_1(args[1])
      
    def postprocess(self, arg):
        total = [arg[0]['a']['b'].sum(), arg[0]['c'].sum(), arg[1].sum()]
        return {'score': sum(total), 'decision': sum(total) > 0}
        

class ModelA(torch.nn.Module):
    def forward(self, args):
        return {'b': torch.randn(args['b'].shape[0], 10)}


class ModelC(torch.nn.Module):
    def forward(self, args):
        return torch.randn(args.shape[0], 10)


class Model1(torch.nn.Module):
    def forward(self, args):
        return torch.randn(args.shape[0], 10)


class Model0(torch.nn.Module):
    def __init__(self, model_a, model_c):
        super().__init__()
        self.model_a = model_a
        self.model_c = model_c

    def forward(self, args):
        return {'a': self.model_a(args['a']), 'c': self.model_c(args['c'])}


model = Main(
    model_0=Model0(
        model_a=ModelA(),
        model_c=ModelC()
    ),
    model_1=Model1()
)
```

We can see that the preprocessor of this model spits out nested elements in a way which the PyTorch `DataLoader` class knows how to deal with:

```
>>> model.preprocess(({'a': {'b': 'orange'}, 'c': 'pear'}, ('apple', 'apple')))
[{'a': {'b': 1}, 'c': 2}, [0, 0]]
```

The batching which the PyTorch `DataLoader` performs over these inputs now fit perfectly into the modular structure of the model, which gives nested output:

```
>>> model.forward(
...     [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
... )
({'a': {'b': tensor([[-2.9543, -1.3461,  1.2037, -1.3606,  0.1646,  0.0240,  0.7850, -1.5709,
             1.2882, -0.4400],
           [-0.2425, -1.1993, -1.1792, -1.3241, -0.8503, -0.5293, -1.6596,  0.4395,
             0.2876, -0.7023]])},
  'c': tensor([[ 1.7177,  0.3557, -0.3067, -1.2059,  0.3755,  0.3976,  0.2048,  0.5670,
            0.4187,  0.1387],
          [ 0.7966, -2.1738, -1.5473, -0.8447,  0.0292, -0.2433, -0.1465, -0.3417,
            0.0497,  0.2354]])},
 tensor([[ 0.2075, -1.2628, -0.3890, -1.1651,  1.5997, -0.3244, -0.8618,  0.7341,
           0.4067,  0.1807],
         [ 0.7603, -0.7152, -0.4594,  0.9974, -0.8822,  0.6915, -1.4039, -0.2845,
           0.5489,  0.1528]]))
```


The problem now is, however, that the batches are now buried deep inside this tree structure. Ideally we'd like to unpack the batch to reflect the serial nature of the input data. This is what the `unpack_batch` function does below.  

```python
from torch.utils import data
import torch
import tqdm


class BasicDataset(data.Dataset):
    def __init__(self, documents, transform=None):
        super().__init__()
        self.documents = documents
        self.transform = transform

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        if self.transform is None:
            return self.documents[item]
        else:
            r = self.transform(self.documents[item])
            return r


def create_batch(args):
    """
    Create a singleton batch in a manner similar to the PyTorch dataloader

    :param args: single data point for batching
    """
    if isinstance(args, (tuple, list)):
        return tuple([create_batch(x) for x in args])
    if isinstance(args, dict):
        return {k: create_batch(args[k]) for k in args}
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, (float, int)):
        return torch.tensor([args])
    raise TypeError('only tensors and tuples of tensors recursively supported...')


def unpack_batch(args):
    """
    Unpack a batch into lines of tensor output.

    :param args: a batch of model outputs

    >>> unpack_batch(torch.randn(1, 10))[0].shape
    torch.Size([10])
    >>> out = unpack_batch([torch.randn(2, 10), torch.randn(2, 3, 5)])
    >>> type(out)
    <class 'list'>
    >>> len(out)
    2
    >>> out = unpack_batch({'a': torch.randn(2, 10), 'b': torch.randn(2, 3, 5)})
    >>> [type(x) for x in out]
    [<class 'dict'>, <class 'dict'>]
    >>> out[0]['a'].shape
    torch.Size([10])
    >>> out[0]['b'].shape
    torch.Size([3, 5])
    >>> out = unpack_batch({'a': {'b': torch.randn(2, 10)}})
    >>> out[0]['a']['b'].shape
    torch.Size([10])
    >>> out[1]['a']['b'].shape
    torch.Size([10])
    """

    if isinstance(args, torch.Tensor):
        return [args[i] for i in range(args.shape[0])]
    else:
        if isinstance(args, list) or isinstance(args, tuple):
            tmp = [unpack_batch(x) for x in args]
            batch_size = len(tmp[0])
            return [[x[i] for x in tmp] for i in range(batch_size)]
        elif isinstance(args, dict):
            tmp = {k: unpack_batch(v) for k, v in args.items()}
            batch_size = len(next(iter(tmp.values())))
            return [{k: v[i] for k, v in tmp.items()} for i in range(batch_size)]
        else:
            raise NotImplementedError


def apply_model(model, args, single=True, verbose=False, **kwargs):
    """
    Apply model to args including pre-processing, forward pass and post-processing.

    :param model: model object including methods *preprocess*, *forward* and *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched) datapoints.
    :param verbose: display progress bar
    :param kwargs: key, value pairs to be passed to dataloader
    """
    if single:
        prepared = model.preprocess(args)
        singleton_batch = create_batch(prepared)
        output = model.forward(singleton_batch)
        if hasattr(model, 'postprocess'):
            return model.postprocess(output)
        return output
    else:
        inputs = BasicDataset(args, model.preprocess)
        loader = data.DataLoader(inputs, **kwargs)
        out = []
        if verbose:
            progress = tqdm.tqdm(total=len(args))
        for batch in loader:
            tmp = model.forward(batch)
            tmp = unpack_batch(tmp)
            if hasattr(model, 'postprocess'):
                tmp = list(map(model.postprocess, tmp))
            out.extend(tmp)
            if verbose:
                progress.update(len(tmp))
        return out
```

Putting everything together we can get end-2-end model outputs using a simple call to `apply_model`:

```
>>> apply_model(
...    model, 
...    ({'a': {'b': 'orange'}, 'c': 'pear'}, ('apple', 'apple')),
...    single=True
... )
{'score': tensor(-3.9864), 'decision': tensor(False)}
```

We can easily switch to batches by toggling `single` off:

```
>>> apply_model(
...     model,
...     [({'a': {'b': 'orange'}, 'c': 'pear'}, ('apple', 'apple')) for _ in range(10)],
...     single=False
... )
[{'score': tensor(0.8145), 'decision': tensor(True)},
 {'score': tensor(-3.7398), 'decision': tensor(False)},
 {'score': tensor(-0.5557), 'decision': tensor(False)},
 {'score': tensor(-4.0278), 'decision': tensor(False)},
 {'score': tensor(-5.3502), 'decision': tensor(False)},
 {'score': tensor(-2.6049), 'decision': tensor(False)},
 {'score': tensor(2.5765), 'decision': tensor(True)},
 {'score': tensor(10.8553), 'decision': tensor(True)},
 {'score': tensor(-4.9206), 'decision': tensor(False)},
 {'score': tensor(4.9929), 'decision': tensor(True)}]
```

Writing models with this type of modular design pattern can lead to a clearer separation of concerns, enforce better practices in management of experimentation, and encourage deep learning developers to write code in a modular and predictable fashion. You can use this function by installing `pip install torchapply`. The source code is [here](https://github.com/blythed/torchapply).

```python
from torchapply import apply_model
```

Equipped with this design pattern, next time we're going to look at how to serialize the whole object in a way which streamlines downstream MLops.
