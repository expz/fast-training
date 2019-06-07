# Inspired by
#   https://github.com/dconathan/madpy-deploy-ml/blob/master/src/main.py

import fire
import model


class Translator(object):
    """A simple translator."""

    def train(self):
        model.train()

    def fren(self, *words):
        fr_text = " ".join(words)
        en_text = model.translate_fren(fr_text)
        print(en_text)


if __name__ == '__main__':
    fire.Fire(Translator)
