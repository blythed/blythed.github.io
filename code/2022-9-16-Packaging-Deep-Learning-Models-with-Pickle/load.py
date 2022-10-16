import pickle

from torchapply import apply_model

with open('model.pkl', 'rb') as f:
    c = pickle.load(f)

print(apply_model(c, 'lorem ipsum'))
print(apply_model(c, ['lorem ipsum', 'lorem ipsum'], single=False, batch_size=2))

