import argparse
import base64
import io
import pickle
from glob import glob

import av  # PyAV is a Pythonic binding for FFmpeg.
from av import VideoFrame
from tqdm import tqdm


def process_videos(path, outpath, filename, sample=4, frameH=224, frameW=224, H=32, W=32):
    """
    - During pre-training, we sparsely sample T = 4 video
    frames and resize them into 224x224 to split into patches
    with H = W = 32. 
    - For all downstream tasks, we adopt the same video frame
    size (224x224) and patch size (32x32) but 5 sparse-sampled
    frames.
    """
    print(f"- Sample Size: {sample}, frame shape: ({frameH} x {frameW}), patch size: ({H} x {W})")
    infile = glob(f'{path.rstrip("/")}/*.mp4')
    pkl = {}    # dictionary {"video_name": encoded frames}
    for video in tqdm(infile):
        v_name = video.split("/")[-1].replace(".mp4", "")
        imgs = get_images(video)
        N = len(imgs)/(sample+1)
        
        pkl[v_name] = []
        for i in range(sample):
            buf = io.BytesIO()
            imgs[int(N*(i+1))].save(buf, format="JPEG") # sparse sample -> select 1 uniformely wrt total time lenght
            pkl[v_name].append(str(base64.b64encode(buf.getvalue()))[2:-1]) # appending into dict a list of encoded frames
    print(f"- Saving dataset: {filename} at path: {outpath}")
    pickle.dump(pkl, open(f"{outpath}/{filename}.pkl", "wb"))
    

def get_images(video):
    imgs = []
    for pack in av.open(video).demux(): # Yields a series of Packet from the given set of Stream
        for buf in pack.decode():       # Send the packet’s data to the decoder and return a list of AudioFrame, VideoFrame or SubtitleSet.
            if type(buf) == VideoFrame:
                imgs.append(buf.to_image().convert("RGB"))
    return imgs

def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, type=str)
    parser.add_argument('--sample', type=int, default=5)
    parser.add_argument('--outpath', '-o', required=True, type=str)
    parser.add_argument('--filename', '-f', required=True, type=str)
    return parser

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    print("- Processing videos...")
    process_videos(
        path=args.path,
        outpath=args.outpath,
        filename=args.filename,
        sample=args.sample,
    )
