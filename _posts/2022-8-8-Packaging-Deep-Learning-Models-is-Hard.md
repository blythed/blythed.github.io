---
layout: post
title: Packaging Deep Learning Models is Hard
---

After you've diligently trained and re-trained your deep neural network and performed model selection over a grid of hyperparameters, as a data-scientist or researcher, your feeling is generally "great now I'm done!". 

*Not so fast!*

It's now time to package and deliver your results to a more realistic testing scenario (e.g. A/B testing), Q&A testing, or to go straight to production via some kind of continuous deployment pipeline. And this is where things can become a huge pain in the posterior.

**Deep learning models expect tensors as inputs and emit tensors as outputs, which are not actionable in any relevant sense.**

In order to actually *use* a deep learning model, we need to define how raw-inputs are pre-processed and tensorial outputs are post-processed.

Why is this a problem?

Well, let's follow the naive approach to solving this problem, and see where we get into difficulties:

1. We save the weights, or the class which encodes the forward pass of our deep-learning model
2. We reload the model in the production environment
3. For all such models, we make sure we always use the same pre-processing and post-processing functions in training and in production

The issue is with step 3. Unless we are dealing with very predictable scenarios, such as classification over a fixed set of labels, this step breaks down. Often the pre-processing and post-postprocessing depend inextricably on the model we trained. For example:

- For a **computer vision** CNN, the pre-processing transformations may not be the same for all networks we train. Some may have inference-time random-ness, or cropping, which others may not.
- For **NLP** models, the tokenizer required for inference may depend on the data which we used in training. This is prudent, since a tokenizer calibrated on training data, will generally yield shorter sequence length, and hence produce more accurate and compact deep-learning models. In such a case, each model will be accompanied by important data dependent artifacts *as part of their pre-processing*.
- In **neural translation**, the post-processing contains key hyper-parameters, such as the number of candidates and search breadth of a beam search inference routine.

Taking stock of these types of cases, we must concede that a fully actionable deep-learning "model" should encapsulate not just the forward pass, but also the logic in the pre- and post- postprocessing. These pieces of logic **should be saved, exported and packaged** together with the forward pass.

And this is where the standard tool kit fails us.

To the best of my knowledge there is no single correct way, enshrined in a code of "best practices for data scientists", which prescribes how to package pre-processing, forward pass and post-processing. In an up-coming series of posts, I'm going to discuss a variety of ways to address this scenario, and their relative strengths and weaknesses, so stay tuned!







