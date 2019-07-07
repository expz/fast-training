import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(APP_DIR)

CONFIG_FILE = os.path.join(PROJECT_DIR, 'config', 'fr2en.yaml')

MODEL_FILE = os.path.join(PROJECT_DIR, 'model', 'fr2en.pth')

TOKENIZER = os.path.join(PROJECT_DIR, 'data', 'moses', 'tokenizer', 'tokenizer.perl')

MAX_LENGTH = int(os.getenv('MAX_LENGTH', 50))

BEAM_WIDTH = int(os.getenv('BEAM_WIDTH', 2))
