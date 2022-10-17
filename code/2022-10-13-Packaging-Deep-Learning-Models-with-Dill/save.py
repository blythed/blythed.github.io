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
    dill.dump(c, f, recurse=True)
