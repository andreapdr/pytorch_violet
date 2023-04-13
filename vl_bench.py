import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import VLBenchDataset
from predict import VIOLET_Foil
import json

import warnings

warnings.filterwarnings("ignore")


def init_model():
    model = VIOLET_Foil()
    model.eval()
    # print(f"- loaded pre-trained VIOLET model")
    return model


def get_dataset(datapath, args):
    return VLBenchDataset(datapath, args)


def run_vlbench(args):
    print(f"- evaluating VIOLET on {args.json_path}")
    device = args.device
    dataset = get_dataset(args.json_path, args)
    model = init_model()
    model.load_ckpt("checkpoints/ckpt_violet_pretrain.pt")
    model.to(device)

    pairwise_accuracy = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    results = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=True if args.debug else False):
            text, video, sample_id, ann_id = batch
            capt, capt_mask = text[0]
            foil, foil_mask = text[1]

            if args.debug:
                print(sample_id[0])

            capt_score = model(
                img=video.to(device), txt=capt.to(device), mask=capt_mask.to(device)
            ).item()

            foil_score = model(
                img=video.to(device), txt=foil.to(device), mask=foil_mask.to(device)
            ).item()

            if capt_score > foil_score:
                pairwise_accuracy += 1

            results[ann_id[0]] = {"scores": [capt_score, foil_score]}

    print(f"- Pairwise Accuracy: {pairwise_accuracy / len(dataset):.3f}")
    json.dump(results, open(f"results_{args.json_path.split('/')[-1]}.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser(description="VIOLET on VL-Bench")
    parser.add_argument(
        "--json_path",
        type=str,
        default="~/datasets/vl-bench/reduced/change-state-action.json",
    )
    parser.add_argument(
        "--video_dir", type=str, default="~/datasets/vl-bench/videos/change-state"
    )
    parser.add_argument("--size_img", type=int, default=224, help="size of the image")
    parser.add_argument("--size_txt", type=int, default=128)
    parser.add_argument(
        "--path_ckpt", type=str, default="./_snapshot/ckpt_violet_pretrain.pt"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_vlbench(args)
