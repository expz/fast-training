import os

APP_DIR = os.getenv('APP_DIR', os.getcwd())

CONFIG_FILE = os.path.join(APP_DIR, 'config', 'fr2en.yaml')

MODEL_FILE = os.path.join(APP_DIR, 'model', 'fr2en.pth')
MODEL_FILE = os.getenv('MODEL_FILE', MODEL_FILE)

TOKENIZER = os.path.join(APP_DIR, 'data', 'moses', 'tokenizer', 'tokenizer.perl')

MAX_LENGTH = int(os.getenv('MAX_LENGTH', 50))

BEAM_WIDTH = int(os.getenv('BEAM_WIDTH', 2))

EXAMPLES_FILE = os.path.join(APP_DIR, 'data', 'examples-fr.txt')
