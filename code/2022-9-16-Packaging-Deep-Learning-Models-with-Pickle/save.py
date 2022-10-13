from package import *
from torchapply import apply_model
import sys
import dill


i = MyTokenizer()
i.calibrate('corpus.txt')
l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
out = MyReader(['good', 'bad', 'ugly'])
c = MyCompoundClass(i, l_, out)
c.eval()

print(apply_model(c, 'lorem ipsum'))
print(apply_model(c, ['lorem ipsum', 'lorem ipsum'], single=False, batch_size=2))
if sys.argv[1] == 'pickle':
    with open('model.pkl', 'wb') as f:
        pickle.dump(c, f)
elif sys.argv[1] == 'dill':
    with open('model.dill', 'wb') as f:
        dill.dump(c, f)
else:
    raise NotImplementedError
