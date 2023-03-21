from collections import Counter
import torch
import av 
import torchvision
from torchvision.transforms import ToPILImage
from av import VideoFrame

from lib import *


class Dataset_Base(T.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.tokzr = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")

    def str2img(self, b):
        img = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        w, h = img.size
        img = TV.transforms.Compose(
            [
                TV.transforms.Pad([0, (w - h) // 2] if w > h else [(h - w) // 2, 0]),
                TV.transforms.Resize([self.args.size_img, self.args.size_img]),
                TV.transforms.ToTensor(),
            ]
        )(img)
        return img

    def str2txt(self, s):
        txt = self.tokzr.encode(
            s, padding="max_length", max_length=self.args.size_txt, truncation=True
        )
        mask = [1 if w != 0 else w for w in txt]
        txt, mask = np.array(txt, dtype=np.int64), np.array(mask, dtype=np.int64)
        return txt, mask


class FoilingDataset(Dataset_Base):
    def __init__(
        self,
        args,
        split,
        tasks=["action_foil", "preState_foil", "postState_foil", "reverse_foil"],
        debug=False,
    ):
        super().__init__(args)

        self.datasetName = args["dataset"]
        self.tasks = ["capt"] + tasks
        if "all" in args["dataset"].lower():
            _allDataset = ["coin", "rareAct", "youCook2"]  # TODO: "smsm"
            self.img = {
                k: pickle.load(open(f"_data/videos/{k}.pkl", "rb")) for k in _allDataset
            }
        else:
            self.img = pickle.load(open(f'_data/videos/{args["dataset"]}.pkl', "rb"))
        self.txt = json.load(open(f'./_data/foils/{args["dataset"]}.json', "r"))[split]
        self.keys = list(self.txt.keys())
        self.debug = False
        if debug:
            self.debug = True
            print("- NB: Running DATASET IN DEBUG MODE (foil sentence is hard-typed")

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
        print(
            f"New view (trigger only={trigger}, task only={_task}) - len: {len(new_list)}"
        )
        self.keys = new_list

    def get_trigger_list(self):
        triggers = [
            self.txt[k]["verb"]
            for k in self.keys
            if self.txt[k]["setting"] == "action recognition (LLM)"
        ]
        counter = self._set_counter(triggers)
        return sorted(
            [(k, v) for k, v in counter.items()], key=lambda x: x[1], reverse=True
        )

    def _set_counter(self, triggers):
        self.counter = Counter(triggers)
        return self.counter

    def __getitem__(self, idx):
        key_dict = self.keys[idx]
        item = self.txt[key_dict]

        img = []
        video_name = (
            f"{item['youtube_id']}_{int(item['start_time'])}_{int(item['end_time'])}"
        )
        if "all" in self.datasetName:
            buffer = self.img[item["original_dataset"]].get(video_name, None)
        elif "smsm" in item["original_dataset"]:
            buffer = self.img.get(video_name.replace("_0_0", ""), None)
        else:
            buffer = self.img.get(video_name, None)
        if buffer is None:
            return None, None, key_dict, item
        for b in buffer:
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        texts = {k: self.str2txt(item[k]) for k in self.tasks}
        return img, texts, key_dict, item


class FoilConcatDataset(T.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.tokzr = self.datasets[0].tokzr

    def get_trigger_list(self):
        for d in self.datasets:
            d.get_trigger_list()
        self.counter = sum(
            [d.counter for d in self.datasets], start=Counter()
        )  # https://stackoverflow.com/questions/30003466/summing-list-of-counters-in-python
        return sorted(
            [(k, v) for k, v in self.counter.items()], key=lambda x: x[1], reverse=True
        )


class NewDataset(Dataset_Base):
    def __init__(self, args):
        super().__init__(args)
        
        self._data = json.load(open(os.path.expanduser(args.benchmark_path)))
        self.data = [d for d in self._data.values()]
        self.instrument = args.instrument
        self.task = args.task
        self.videodir = os.path.expanduser(args.videodir)
        self.sample = 5
        print(f"- evaluating instrument: {self.instrument} on setting: {self.task}")

    
    def __len__(self):
        return len(self.data)
    
    def _get_text(self, idx):
        capt, capt_mask = self.str2txt(self.data[idx]["foils"][self.task][0])
        foil, foil_mask = self.str2txt(self.data[idx]["foils"][self.task][1])
        return (self._convert(capt), self._convert(capt_mask)), (self._convert(foil), self._convert(foil_mask))
    
    def _convert(self, d):
        d = torch.tensor(d)
        d = d.unsqueeze(0)
        return d
    
    def _get_video(self, idx):
        video_path = os.path.join(self.videodir, self.data[idx]["video-id"]) + ".mp4"
        start_time = self.data[idx]["timestamp"][0]
        end_time = self.data[idx]["timestamp"][1]
        sampled_frames = self._process_video(video_path, start_time, end_time)
        imgs = []
        for s in sampled_frames:
            s = s.permute(2, 0, 1)
            s = ToPILImage()(s).convert("RGB")
            imgs.append(self._vlbench_str2img(s))
        # img = T.cat(imgs, dim=0)
        img = T.stack(imgs) # TODO
        return img.unsqueeze(0)

    def __getitem__(self, idx):
        text = self._get_text(idx)
        video = self._get_video(idx)
        return text, video

    def _process_video(self, video_path, start_time, end_time):
        sampled_frames = self.get_images(video_path, start_time, end_time)
        return sampled_frames
    
    def _vlbench_str2img(self, img):
        w, h = img.size
        img = TV.transforms.Compose(
            [
                TV.transforms.Pad([0, (w - h) // 2] if w > h else [(h - w) // 2, 0]),
                TV.transforms.Resize([self.args.size_img, self.args.size_img]),
                TV.transforms.ToTensor(),
            ]
        )(img)
        return img

    def get_images(self, video_path, start_time, end_time):
        """
        - During pre-training, we sparsely sample T = 4 video
        frames and resize them into 224x224 to split into patches
        with H = W = 32. 
        - For all downstream tasks, we adopt the same video frame
        size (224x224) and patch size (32x32) but 5 sparse-sampled
        frames.
        
        imgs = []
        for pack in av.open(f).demux():
            for buf in pack.decode():
                if str(type(buf))=="<class 'av.video.frame.VideoFrame'>":
                    imgs.append(buf.to_image().convert('RGB'))
        N = len(imgs)/(args.sample+1)
        
        pkl[vid] = []
        for i in range(args.sample):
            buf = io.BytesIO()
            imgs[int(N*(i+1))].save(buf, format='JPEG')
            pkl[vid].append(str(base64.b64encode(buf.getvalue()))[2:-1])

        """
        sampled_frames = []
        video = torchvision.io.read_video(video_path, pts_unit="sec", start_pts=start_time, end_pts=end_time)[0]
        N = video.shape[0]/(self.sample+1)

        for i in range(1, self.sample-2):
            sampled_frames.append(video[int(N*(i+1))])            
        
        # forcing fist and last frame
        sampled_frames.insert(0, video[0])
        sampled_frames.append(video[-1])

        return sampled_frames
    