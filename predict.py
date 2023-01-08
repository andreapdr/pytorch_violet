import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from transformers import logging

from lib import *
from model import VIOLET_Base
from dataset import FoilingDataset, FoilConcatDataset
from utils import convert_to_string, task2str

logging.set_verbosity_error()


class VIOLET_Foil(VIOLET_Base):
    """
    Violet Class for foil prediction. 
    Computes the matching between sentence and image/video signal via
    CrossModal Transformer (CT, Section 3.2 + VTM).
    Initialzing only FC_VTM (self.fc) head.
    - VT (init from: Video-Swin Transformer base):
      it has 1 [CLS_V] token in front of each patch -> output: (1+_h*_w*)*_T
    - LT (init from: Bert-base):  
    - CT (init from: Bert-base):
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(*[
                            nn.Dropout(.1),
                            nn.Linear(768, 768*2),
                            nn.ReLU(inplace=True),
                            nn.Linear(768*2, 1)
                            ])
    
    def forward(self, img, txt, mask):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W // 32           # patch size (_H=224 // 32 and _W=224 // 32 )
        _O = min(_B, 4)                     # min between (batchSize and 4) -> controls the number of negative contrastive captions for VTM matching

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out_vtm = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _O]) / 0.05          # select the embedding at [CLS] token index across 
                                                                                            # allbatches -> out[:, CLS_index, :]
                                                                                            # NB: CLS INDEX = (1+_h*_w)*_T  --> it is +1 b/c they have a CLS_V token in front of each video frame https://github.com/tsujuifu/pytorch_violet/issues/4                                                                        
                                                                                            # TODO: why / 0.05 ? to avoid underflow?

        return out_vtm
    

if __name__=='__main__':
    print('- Running VIOLET foil evaluation...')
    ARGS_PATH = '_config/args_multi3bench.json'
    args = json.load(open(ARGS_PATH, 'r'))
    print(f"- Dataset: {args['dataset']}")
    MASK_VIDEO = False
    TASKS = ["action_foil"]
    if MASK_VIDEO:
        print(f"WARNING: discarding video modality - performing ablation study!")
    split = 'test'
    dataset = FoilingDataset(args, "test", tasks=TASKS, debug=False)
    json_res = json.load(open(f"_data/foils/{args['dataset']}.json"))
    tokenizer = dataset.tokzr

    model = VIOLET_Foil().cuda()
    model.load_ckpt(args["path_ckpt"]) 
    
    model.eval()
    results = []
    with torch.no_grad():
        for i, (img, texts, data_id, metadata) in enumerate(tqdm(dataset)):
            if img is None:
                print(f"- (warning) ({i}), id: {data_id} is None")
                continue

            if MASK_VIDEO:
                img = torch.zeros_like(img)

            for task in (["capt"] + TASKS):
                input_text = torch.tensor(texts[task][0]).unsqueeze(0).cuda()
                input_mask = torch.tensor(texts[task][1]).unsqueeze(0).cuda()
                out = model(img.unsqueeze(0).cuda(), input_text, input_mask).squeeze().detach().cpu().numpy().item()
                json_res["test"][data_id][f"VIOLET_{task}"] = out
            
    now = f"{datetime.now():%m%d%H%M}"
    json.dump(json_res, open(f"foilResults/results_multi3bench_{now}.json", "w"))
    exit("Done!")