---
layout: page
title: About
permalink: /about/
---

I'm an AI researcher, entrepreneur and architect. I've published extensively on the topic of [machine learning and AI](https://scholar.google.com/citations?user=-H7cJ8wAAAAJ&hl=en&oi=ao), including some of the [most widely cited work](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=-H7cJ8wAAAAJ&citation_for_view=-H7cJ8wAAAAJ:WF5omc3nYNoC) in natural language processing of the last few years, made major initial contributions to the open source package [Flair](https://github.com/flairNLP/flair) (now over 10,000 stars), was a driving force behind Zalando's [AI ramp up and NLP strategy](https://corporate.zalando.com/en/company/research-zalando), bootstrapped a 2-man startup [LF1](https://lf1.io/) to 7-figure revenue within 1.5 years, and [exited](https://www.attraqt.com/resources/attraqt-acquires-ai-ip-assets-from-aleph-one/#:~:text=Thursday%20October%201%2C%202020%E2%80%A6,assets%20from%20Aleph%20One%20GmbH.) the deep-learning powered e-commerce search suite [Aleph-Search](https://www.alephsearch.com/) to the market leader in e-commerce search and navigation, [Attraqt Group PLC](https://www.attraqt.com/). 

If you are interested in working with me on a speculative project involving AI, deep-learning, natural language processing, semantic information retrieval or related areas, then please [reach out](#contact-me) with a short description of your idea and why the project is important to your company or organization.

### Education

- *M.Math.Phil* (1st class) in Mathematics and Philosophy from [Oxford University, UK](https://www.ox.ac.uk/) in 2007
- *M.Sc* in Computational Neuroscience and Machine Learning from the [Bernstein Centre for Computational Neuroscience, Berlin, Germany](https://www.bccn-berlin.de/) in 2009
- *Ph.D* in Computer Science (*Summa cum Laude*) from the [Technical University of Berlin, Germany](https://www.ml.tu-berlin.de/menue/machine_learning/) in 2015

### Employment

- Tutor for applied mathematics at [African Institute for Mathematical Sciences](https://nexteinstein.org/), Bagamoyo, Tanzania (2015-2016)
- Postdoctoral researcher in [Machine Learning and Statistics](https://www.dzne.de/forschung/forschungsbereiche/populationsforschung/forschungsgruppen/mukherjee/curriculum-vitae/) group at German Institute for Neurodegenerative Disease ([DZNE](https://www.dzne.de/)) (Helmholtz), Bonn, Germany (2016-2017) 
- Research scientist at [Zalando Research](https://corporate.zalando.com/en/company/research-zalando) in natural language and deep learning sub-group (2017-2018)
- Co-founder and CEO of [LF1](https://lf1.io/) as 33% equal share-holder (2019-)
- [Independent AI researcher, entrepreneur and architect](https://blythed.github.io/) (2022-)

### Projects

#### Aleph Search

[Aleph Search](https://www.alephsearch.com/) is a complete suite for e-commerce search and navigation powered end-2-end with deep-learning, which I developed jointly with [Alexander Schlegel](https://www.linkedin.com/in/alexander-schlegel-32372aa9/?originalSubdomain=de) as the first commercial project at [LF1](https://lf1.io/). Aleph Search was [acquired](https://www.attraqt.com/resources/attraqt-acquires-ai-ip-assets-from-aleph-one/#:~:text=Thursday%20October%201%2C%202020%E2%80%A6,assets%20from%20Aleph%20One%20GmbH.) by Attraqt Group PLC in 2020. I continued the journey with Attraqt in an advisory capacity until 2022, during which time Aleph Search (now Attraqt's "[AI powered search](https://www.attraqt.com/wp-content/uploads/2022/02/ai-powered-search.pdf)") was deployed to production on the websites of the major e-commerce players in Europe. Among these are:

- Adidas
- Asos
- Missguided
- Pretty Little Thing
- Screwfix
- Secret Sales
- Superdry
- Waitrose

![]({{ site.baseimg }}/images/alephsearch.png)

Aleph Search comprises:

- semantic search (search with "meaning")
- similar product recommendation (search for similar "looking" products)
- reverse image-search (find a product using an uploaded image of worn fashion items)
- shop-the-look (find all items in an image containing multiple fashion items)
- auto-complete (suggest searches based on partial search)
- product tagging (fill out missing, or additonal product information)

#### Contextual string embeddings

In a research project together with [Alan Akbik](https://alanakbik.github.io/) and [Roland Vollgraf](https://www.linkedin.com/in/rolandvollgraf?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=de), we demonstrated in [the first paper of its kind](https://aclanthology.org/C18-1139/?utm_campaign=piqcy&utm_medium=email&utm_source=Revue%20newsletter), that pretraining a high quality language model at the character level, leads to highly cost effective representations for named-entity recognition and general word-level sequence labelling tasks. Our analysis showed that character level language modelling combines the best of sub-word level granularity with meaningful semantic vectorial neighbourhoods for a number of downstream tasks. In languages such as German, which has a highly compositional structure from sub-word to word level, this leads to improvements over the SOTA of over 10%. The research paper was published and presented at [COLING 2018](https://coling2018.org/), and made a major contribution to the proliferation of language model representations such as [Bert](https://en.wikipedia.org/wiki/BERT_(language_model)), which form an integral part of e.g. the Google search algorithm.

![]({{ site.baseimg }}/images/neuralLMtagging.png)

#### Flair

![]({{ site.baseimg }}/images/flair.png){: width="80" }

I was an initial contributor to the [Flair](https://github.com/flairNLP/flair) library, an initiative of [Alan Akbik](https://alanakbik.github.io/) (now Humboldt University) at Zalando Research. Flair is a simple but effective library for core NLP tasks, making it straightforward to deploy and combine word level embeddings for a variety of tasks, including:

- named entity recognition (NER)
- part-of-speech tagging (PoS)
- sense disambiguation
- sentence level classification

#### PADL

[Pipeline Abstractions for Deep Learning (PADL)](https://padl.ai/), codifies the approach we took to development at LF1 and is a project I initiated together with colleague and [LF1](https://lf1.io/) co-founder [Alexander Schlegel](https://www.linkedin.com/in/alexander-schlegel-32372aa9/?originalSubdomain=de).

Just as programs are read more often than they are written, so are deep learning models used more often than they are trained. The PyTorch ecosystem has many tools for training models. However, this is only the beginning of the journey. Once a model has been trained, it will be shared, and used in a multitude of contexts, often on a daily basis, in operations, evaluation, comparision and experimentation by data scientists. The *use* of the trained model, is how value is extracted out of its weights. Despite this important fact, support for using deep-learning models up to now has been very thin in the PyTorch ecosystem and beyond. PADL is a tool which fills this void.

![]({{ site.baseimg }}/images/padl_schematic.png){: height="600" }

#### Prediction and quantification of athletic performance

Together with [Franz Kiraly](https://scholar.google.de/citations?user=VYi_04kAAAAJ&hl=de), I developed and [published a new approach](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0157257) to predicting and quantifying athletic performance, based on a novel combinatorial-matrix completion algorithm. The algorithm outperforms state-of-the-art approaches based on e.g. nuclear norm minimization as well as classic scoring methods such as the widely used "[Riegel's formula](https://www.runnersworld.com/uk/training/a761681/rws-race-time-predictor/)". Integral to the approach is the assumption that distinct athletes weight short-distance, middle-distance and long-distance prowess relatively as a result of physiological make-up, training state and gender; the algorithm assesses these relative strengths through performance over a range of distances, and extrapolating to the desired distance by triangulating the athlete in a space of "prototypical" athletes.

![]({{ site.baseimg }}/images/athletic_performance.png)

### Current interests

At the moment I am interested in these things (among a great deal of other things):

- Latest NLP and deep-learning research.
- In data-base computation in the context of semantic information retrieval.
- Applications of deep-learning to Web3.
- MLops, particularly as regards pipelines taking models from research and experimentation all the way through to production.

## Technology stack

- Python 2, 3
- PyTorch
- Tensorflow
- Git
- PostGreSQL
- MongoDB
- Flask
- AWS
  - EC2
  - Cloudformation
  - s3
  - IAM
  - AutoScaling
- Docker
- Jenkins
- Javascript
- C/ C++

## Machine learning stack

- Classical machine learning
  - Kernel methods
  - Source separation
  - Clustering
  - Dimension reduction
- Deep learning
  - Convolution networks for vision (e.g. ResNet architectures)
  - Deep ranking/ representation learning (e.g. Siamese network learning)
  - Bounding box regression (e.g. Yolo architectures)
  - Image segmentation (e.g. UNet)
  - Language modelling (e.g. transformers and RNNs)
  - Sequence-2-sequence learning (e.g. transformers and RNNs)
  - Image generation with generative adversarial networks (e.g.StyleGAN+)

## Useful links

- [Google scholar profile](https://scholar.google.de/citations?user=-H7cJ8wAAAAJ&hl=en)
- [GitHub profile](https://github.com/blythed)
- [Medium profile](https://medium.com/@duncanblythe)
- [LinkedIn profile](https://www.linkedin.com/in/duncan-blythe-71877312b/)
- [LF1 website](https://lf1.io/)
- [PADL project website](https://padl.ai/)

### Contact me

[duncan (no-space) blythe (at) gmail (dot) com]()