---
layout: post
title: Conventions for packaging deep learning models
---

We talked in the [last post](https://blythed.github.io/Packaging-Deep-Learning-Models-is-Hard/) about the need for a code of "best practices" for packaging and exporting deep learning models. In this post, I'm going to propose some handy conventions, which I believe will assist in the interoperability between deep learning models and the broader ecosystem.

Let's imagine that we've saved a computer vision model for classification, and for the sake of argument, assume this was trained in PyTorch - `torch`. Then in a typical case, we'd first reboot the model using `torch.load` and / or `self.load_state_dict`. Having done this, we'll want to check that the model is really the one that we trained, by, for instance, passing an example input into the model and checking we get what we had expected. For a computer vision model, this means loading an image, for instance with `PIL.Image.open` and then using `torchvision` to convert the image to a tensor. But this is not yet sufficient, we'd typically also need to perform some standardization of the input using certain normalization constants. In many cases, the normalization comes from the [Image-Net training set](https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/8). These "magic numbers", namely `mean=(0.485, 0.456, 0.406)` and `std=(0.229, 0.224, 0.225)` need to be stored and reloaded in order for the tensor input to make sense for the network. Finally a batch of one tensor needs to be constructed to run the forward pass over the input image. We can see that there are potentially many delicate steps here which should be recorded in as robust and reproducible manner as possible.

The next step is simply to pass the batch to the `forward` method of the `torch.nn.Module`, getting typically (a) tensor(s) of logits. The battle is not yet won however, we still have work to do. That is, we need to normalize the predictions to get probabilities, then order these probabilities, and threshold to a sensible confidence to get our predictions. In order to make these predictions actionable, we'd need to load a dictionary of human- or program-readable labels, so that each probability is paired with the name of the class for which the probability is valid, facilitating downstream tasks and processing.

These are the typical steps for applying our network to an input and making sense of an output. In some scenarios however, we'll want to apply these steps to multiple or streaming inputs, obtaining many outputs. For these cases, we'd want to construct a `torch.utils.data.DataLoader` instance in order to get fast processing, and then map the postprocessing over the outputs.

Based on these considerations, here is the minimal required object structure for a PyTorch model, in order to perform these tasks:

```python
class MyModel(torch.nn.Module):
  
    def forward(self, *args, **kwargs):
      	... # differentiable parts of neural network here
       
    def preprocess(self, *args, **kwargs):
        ... # method which defines how to convert a raw input to a tensor
        
    def postprocess(self, *args, **kwargs):
        ... # method which defines how to convert a single output into 
            # a useful human-readable format
      
```

How would we use this in practice?

For a single data point `sample` we would execute something like:

```python
tensor_ = model.preprocess(x=sample)
batch = tensor_.unsqueeze(0)
output = model.forward(batch)
predictions = model.postprocess(output[0])
```

For predictions over data points we would use a generic `torch.utils.data.DataSet`:

```python
class DataSetWrapper(torch.utils.data.DataSet):

    def __init__(self, model, datapoints):
        self.model = model
        self.datapoints = datapoints
        
    def __len__(self):
        return len(self.datapoints)
        
    def __getitem__(self, *args, **kwargs):
        return self.model.preprocess(*args, **kwargs)
```

Then we would do:

```python
dataloader = torch.utils.data.DataLoader(
    DataSetWrapper(model, datapoints),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

outputs = []
for batch in dataloader:
    predictions = model.forward(**batch)
    outputs.append(predictions)
    
outputs = [x[0] for x in torch.cat(outputs, 0)]
hr_outputs = list(map(outputs, model.postprocess))
```

These code snippets could easily be adapted to handle streaming data as well as a range of practical use cases. For instance, this design pattern is very close to the [design pattern required by TorchServe](https://pytorch.org/serve/custom_service.html#writing-a-custom-handler-from-scratch-for-prediction-and-explanations-request).

The final challenge is to find a way to save our objects in training to include serialization of the `preprocess` and `postprocess` methods. In the next few posts we're going to look at these methods:

- Pickle
- Dill
- PADL
- AI-json

So stay tuned!

