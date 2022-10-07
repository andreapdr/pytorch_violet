import json
import pickle
from collections import Counter

from dataset import Dataset_Base


def convert_to_string(tokenizer, txt):
    sent = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(txt)).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    return sent.lstrip(" ").rstrip(" ")


class FoilingDataset(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args)
        
        self.img = pickle.load(open(f'_data/videos/{args["dataset"]}/encoded/{args["dataset"]}.pkl', 'rb'))
        self.txt = json.load(open(f'./_data/foils/txt_{args["dataset"]}.json', 'r'))[split]
        self.keys = list(self.txt.keys())

    def __len__(self):
        return len(self.keys)
    
    def set_view(self, trigger=None, task=None):
        new_triggers = []
        new_tasks = []
        if trigger is not None:
            for k in self.keys:
                if self.txt[k]["verb"] == trigger:
                    new_triggers.append(k)
                if task is not None:
                    if self.txt[k]["linguistic_phenomena"] == task:
                        new_tasks.append(k)
        if task is None:
            new_tasks = self.keys
        
        new_list = list(set(new_triggers).intersection(set(new_tasks)))
        _task = "all" if task is None else task
        print(f'New view (trigger only={trigger}, task only={_task}) - len: {len(new_list)}')
        self.keys = new_list
    
    def get_trigger_list(self):
        triggers = [self.txt[k]["verb"] for k in self.keys]
        counter = Counter(triggers)
        return sorted([(k,v) for k,v in counter.items()], key=lambda x: x[1], reverse=True)

    def __getitem__(self, idx):
        key_dict = self.keys[idx]
        item = self.txt[key_dict]
        video_id = item["youtube_id"]
        
        img = []
        video_name = f"{item['youtube_id']}_{int(item['start_time'])}_{int(item['end_time'])}"
        buffer = self.img.get(video_name, None)
        if buffer is None:
            return None, None, key_dict, video_id
        for b in buffer: 
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        texts = (self.str2txt(item['caption'].replace("<", "").replace(">", "")), self.str2txt(item['foil'].replace("<", "").replace(">", "")))
        return img, texts, key_dict, video_id
