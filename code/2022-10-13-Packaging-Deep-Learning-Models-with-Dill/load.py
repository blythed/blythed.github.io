import dill

from torchapply import apply_model

with open('model.dill', 'rb') as f:
    c = dill.load(f, ignore=True)

print(apply_model(c, 'lorem ipsum'))
