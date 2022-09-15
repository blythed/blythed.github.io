import io
import pickle
import sentencepiece
import sys
import torch


class MyTokenizer:
    def __init__(self, vocab_size=100):
        self.model = io.BytesIO()
        self.vocab_size = vocab_size

    def calibrate(self, file):
        with open(file, 'r') as f:
            sentencepiece.SentencePieceTrainer.train(
                sentence_iterator=iter(f.readlines()),
                model_writer=self.model,
                vocab_size=self.vocab_size,
            )
        self.tokenizer =  sentencepiece.SentencePieceProcessor(model_proto=self.model.getvalue())

    def __call__(self, x):
        return self.tokenizer.encode(x)


class MyReader:
    def __init__(self, d):
        self.d = d

    def __call__(self, x):
        pred = x.topk(1)[1].item()
        return self.d[pred]


class MyLayer(torch.nn.Module):
    def __init__(self, n_symbols, n_classes, n_embed=64, n_hidden=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, n_embed)
        self.rnn = torch.nn.GRU(n_embed, n_hidden, 1, batch_first=True)
        self.proj = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        return self.proj(self.rnn(self.embedding(x))[0])


class MyCompoundClass:
    def __init__(self, tokenizer, layer, decoder):
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder

    def __call__(self, x):
        encoded = self.tokenizer(x)
        tensor_ = torch.LongTensor(encoded)[None, :]
        print('loading with pickle...')
        pred = self.layer(tensor_)[0, -1]
        output = self.decoder(pred)
        return output


if __name__ == '__main__':
    if sys.argv[1] == 'load':

        with open('model.pkl', 'rb') as f:
            i = pickle.load(f)
        print(i('lorem ipsum'))

    elif sys.argv[1] == 'save':

        i = MyTokenizer()
        i.calibrate('corpus.txt')
        l_ = MyLayer(i.tokenizer.get_piece_size(), 3)
        out = MyReader(['good', 'bad', 'ugly'])
        c = MyCompoundClass(i, l_, out)
        print(c('lorem ipsum'))
        with open('model.pkl', 'wb') as f:
            pickle.dump(c, f)

    else:
        raise NotImplementedError
