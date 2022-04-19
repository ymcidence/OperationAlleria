import os

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
DATA_PATH = os.path.join(ROOT_PATH, 'data')

# os.environ["TFHUB_CACHE_DIR"] = os.path.join(ROOT_PATH, 'data', 'pretrained')
