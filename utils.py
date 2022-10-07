import json
import pickle
from collections import Counter

from dataset import Dataset_Base


def convert_to_string(tokenizer, txt):
    sent = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(txt)).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    return sent.lstrip(" ").rstrip(" ")


