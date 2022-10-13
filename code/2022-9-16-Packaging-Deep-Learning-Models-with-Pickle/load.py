import dill
import pickle
import sys

from torchapply import apply_model

if sys.argv[1] == 'pickle':
    with open('model.pkl', 'rb') as f:
        c = pickle.load(f)
elif sys.argv[1] == 'dill':
    with open('model.dill', 'rb') as f:
        c = dill.load(f)
else:
    raise NotImplementedError

print(apply_model(c, 'lorem ipsum'))
print(apply_model(c, ['lorem ipsum', 'lorem ipsum'], single=False, batch_size=2))

