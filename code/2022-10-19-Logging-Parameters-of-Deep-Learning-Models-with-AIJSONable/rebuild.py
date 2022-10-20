from aijson.build import build


d = {
  "var0": {
    "module": "mymodels",
    "caller": "MyTokenizer",
    "kwargs": {
      "calibration_file": "corpus.txt",
      "vocab_size": 100
    }
  },
  "var1": {
    "module": "mymodels",
    "caller": "MyLayer",
    "kwargs": {
      "n_symbols": 100,
      "n_classes": 3
    }
  },
  "var2": {
    "module": "mymodels",
    "caller": "MyReader",
    "kwargs": {
      "d": [
        "good",
        "bad",
        "ugly"
      ]
    }
  },
  "var3": {
    "module": "mymodels",
    "caller": "MyCompoundClass",
    "kwargs": {
      "tokenizer": "$var0",
      "layer": "$var1",
      "decoder": "$var2"
    }
  }
}


out = build(d)

print(out)