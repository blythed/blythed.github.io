from package import *
from torchapply import apply_model
import pickle


i = MyTokenizer()
i.calibrate('corpus.txt')
l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
out = MyReader(['good', 'bad', 'ugly'])
c = MyCompoundClass(i, l_, out)
c.eval()

print(apply_model(c, 'lorem ipsum'))
print(apply_model(c, ['lorem ipsum', 'lorem ipsum'], single=False, batch_size=2))
with open('model.pkl', 'wb') as f:
    pickle.dump(c, f)
