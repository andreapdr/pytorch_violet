import torch
import torchvision
from torchvision.transforms import ToPILImage
from math import floor, ceil

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


class VLBenchDataset(Dataset_Base):
    def __init__(self, datapath, args):
        super().__init__(args)
        
        self._data = json.load(open(os.path.expanduser(datapath)))
        self.data = [d for d in self._data.values()]
        self.videodir = os.path.expanduser(args.video_dir)
        self.sample = 5
    
    def __len__(self):
        return len(self.data)
    
    def _get_text(self, idx):
        _capt = self.data[idx]["caption"]
        _foil = self.data[idx]["foils"][0]
        capt, capt_mask = self.str2txt(_capt)
        foil, foil_mask = self.str2txt(_foil)
        return (self._convert(capt), self._convert(capt_mask)), (self._convert(foil), self._convert(foil_mask))
    
    def _convert(self, d):
        d = torch.tensor(d)
        return d
    
    def _get_video(self, idx):
        if self.data[idx]["youtube_id"] is not None:
            video_fname = self.data[idx]["youtube_id"] + ".mp4"
        else:
            video_fname = self.data[idx]["video_file"]
            if self.data[idx]["dataset"] == "something-something-v2":
                video_fname += ".webm"
            elif self.data[idx]["dataset"] == "ikea_asm":
                video_fname += ".avi"
            else:
                video_fname += ".mp4"
                
        video_path = os.path.join(self.videodir, video_fname)
        start_time = self.data[idx]["start_time"]
        end_time = self.data[idx]["end_time"]
        sampled_frames = self._process_video(video_path, start_time, end_time)
        imgs = []
        for s in sampled_frames:
            s = s.permute(2, 0, 1)
            s = ToPILImage()(s).convert("RGB")
            imgs.append(self._vlbench_str2img(s))
        img = T.stack(imgs)
        return img

    def __getitem__(self, idx):
        text = self._get_text(idx)
        video = self._get_video(idx)
        sample_id = self.data[idx]["dataset_idx"]
        return text, video, sample_id, str(idx)

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
        if start_time is None:
            video = torchvision.io.read_video(video_path, pts_unit="sec")[0]
        else:
            video = torchvision.io.read_video(video_path, pts_unit="sec", start_pts=floor(start_time), end_pts=ceil(end_time))[0]
        N = video.shape[0]/(self.sample+1)

        for i in range(1, self.sample-2):
            sampled_frames.append(video[int(N*(i+1))])            
        
        # forcing fist and last frame
        sampled_frames.insert(0, video[0])
        sampled_frames.append(video[-1])

        return sampled_frames
    