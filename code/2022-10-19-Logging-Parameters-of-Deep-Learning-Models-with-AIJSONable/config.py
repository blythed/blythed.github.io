from mymodels import *

from aijson import logging_context
import json

with logging_context() as lc:
    i = MyTokenizer(calibration_file='corpus.txt', vocab_size=100)
    l_ = MyLayer(n_symbols=i.tokenizer.get_piece_size(), n_classes=3)
    out = MyReader(d=['good', 'bad', 'ugly'])
    c = MyCompoundClass(tokenizer=i, layer=l_, decoder=out)

print(json.dumps(lc, indent=2))