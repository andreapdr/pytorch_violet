from collections import Counter
from math import floor, ceil

from lib import *


class Dataset_Base(T.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.tokzr = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    def str2img(self, b):
        img = Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
        w, h = img.size
        img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                     TV.transforms.Resize([self.args['size_img'], self.args['size_img']]), 
                                     TV.transforms.ToTensor()])(img)
        return img
    
    def str2txt(self, s):
        txt = self.tokzr.encode(s, padding='max_length', max_length=self.args['size_txt'], truncation=True)
        mask = [1 if w!=0 else w for w in txt]
        txt, mask = np.array(txt, dtype=np.int64), np.array(mask, dtype=np.int64)
        return txt, mask
        

class FoilingDataset(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args)
        
        self.img = pickle.load(open(f'_data/videos/{args["dataset"]}.pkl', 'rb'))
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
        counter = self._set_counter(triggers)
        return sorted([(k,v) for k,v in counter.items()], key=lambda x: x[1], reverse=True)

    def _set_counter(self, triggers):
        self.counter = Counter(triggers)
        return self.counter

    def __getitem__(self, idx):
        key_dict = self.keys[idx]
        item = self.txt[key_dict]
        video_id = item["youtube_id"]
        
        img = []
        metadata = {
            "original_caption": item["original_caption"],
            "action": item["verb"],
            "video_id": video_id,
            "dataset_id": key_dict
            }
        video_name = f"{item['youtube_id']}_{int(item['start_time'])}_{int(item['end_time'])}"
        buffer = self.img.get(video_name, None)
        if buffer is None:
            return None, None, metadata
            # return None, None, key_dict, video_id, item['original_caption']
        for b in buffer: 
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        texts = (self.str2txt(item['caption'].replace("<", "").replace(">", "")), self.str2txt(item['foil'].replace("<", "").replace(">", "")))
        return img, texts, metadata
        # return img, texts, key_dict, video_id, item['original_caption']


class FoilConcatDataset(T.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.tokzr = self.datasets[0].tokzr

    def get_trigger_list(self):
        for d in self.datasets:
            d.get_trigger_list()
        self.counter = sum([d.counter for d in self.datasets], start=Counter())         # https://stackoverflow.com/questions/30003466/summing-list-of-counters-in-python
        return sorted([(k, v) for k, v in self.counter.items()], key=lambda x: x[1], reverse=True)
