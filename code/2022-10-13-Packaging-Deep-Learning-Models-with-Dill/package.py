import io
import sentencepiece
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
        # print('testing tokenizer!')
        return self.tokenizer.encode(x)


class MyReader:
    def __init__(self, d):
        self.d = d

    def __call__(self, x):
        pred = x.topk(1)[1].item()
        return {'decision': self.d[pred], 'prediction': x.tolist()}


class MyLayer(torch.nn.Module):
    def __init__(self, n_symbols, n_classes, n_embed=64, n_hidden=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_symbols, n_embed)
        self.rnn = torch.nn.GRU(n_embed, n_hidden, 1, batch_first=True,
                                dropout=0)
        self.proj = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        return self.proj(self.rnn(self.embedding(x))[0])


class MyCompoundClass(torch.nn.Module):
    def __init__(self, tokenizer, layer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.layer = layer
        self.decoder = decoder
        self.function = lambda x: x + 2

    def preprocess(self, arg):
        return torch.tensor(self.tokenizer(arg))

    def forward(self, args):
        output = self.layer(args)[:, -1, :]
        # print('testing testing 123')
        return self.function(output)

    def postprocess(self, x):
        return self.decoder(x)


if __name__ == '__main__':
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
