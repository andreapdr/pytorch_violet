import torch
from argparse import ArgumentParser
from dataset import NewDataset
from predict import VIOLET_Foil

import warnings
warnings.filterwarnings("ignore")


def init_model():
    model = VIOLET_Foil()
    model.eval()
    print(f"- loaded pre-trained VIOLET model")
    return model

def get_dataset(args):
    return NewDataset(args)

def run_vlbench(args):
    device = args.device
    dataset = get_dataset(args)
    model = init_model()
    model.load_ckpt("checkpoints/ckpt_violet_pretrain.pt")
    model.to(device)

    pairwise_accuracy = 0
    with torch.no_grad():
        for text, video in dataset:
            capt, capt_mask = text[0]
            foil, foil_mask = text[1]

            capt_score = model(
                img=video.to(device),
                txt=capt.to(device),
                mask=capt_mask.to(device)
            ).item()

            foil_score = model(
                img=video.to(device),
                txt=foil.to(device),
                mask=foil_mask.to(device)
            ).item()

            if capt_score > foil_score:
                pairwise_accuracy += 1
    
    print(f"- Pairwise Accuracy: {pairwise_accuracy / len(dataset):.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="VIOLET on VL-Bench")
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="~/datasets/vl-bench/cos-balanced.reduced.json",
    )
    parser.add_argument("--size_img", type=int, default=224, help="size of the image")
    parser.add_argument("--size_txt", type=int, default=128)
    parser.add_argument(
        "--path_ckpt", type=str, default="./_snapshot/ckpt_violet_pretrain.pt"
    )
    parser.add_argument("--instrument", type=str, default="change-of-state")
    parser.add_argument("--task", type=str, default="action")
    parser.add_argument("--videodir", type=str, default="~/datasets/vl-bench/videos")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_vlbench(args)
