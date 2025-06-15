import sys
sys.path.append('..')
import logging
import os
import sys
import tqdm
import glob
import random

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from npy_append_array import NpyAppendArray

from dataclasses import dataclass, asdict
    
from src.modules import (
    ResynthesisModule, 
    MimiResynthesisModule,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_content_feature")
random.seed(1234)


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(root, nshard, rank):
    audio_files = glob.glob(f"{root}/**/*.wav", recursive=True)
    audio_files = random.sample(audio_files, len(audio_files)//10) # Randomly sample 10% of the files

    start, end = get_shard_range(len(audio_files), nshard, rank)
    audio_files = audio_files[start:end]
    def iterate():
        for file in audio_files:
            yield f"{file}"
    return iterate, len(audio_files)


def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            # print(feat.shape)
            feat = np.ascontiguousarray(feat, dtype=np.float32)
            feat_f.append(feat)
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


class ContentFeatureReader(object):
    def __init__(self, model, sample_rate=16000):
        self.model = model.cuda()
        self.sample_rate = sample_rate
        logging.info(f"model loaded")

    def read_audio(self, path, ref_len=None):
        wav = librosa.load(path, sr=self.sample_rate)[0]
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.inference_mode():
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda()
            feat = self.model.encoder.net(x)
            if isinstance(feat, tuple):
                feat = feat[1]
            feat = feat.transpose(1, 2)
        return feat.squeeze(0).cpu().numpy()


def main(root, split, model, nshard, rank, feat_dir):
    reader = ContentFeatureReader(model)
    generator, num = get_path_iterator(root, nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("tsv_dir")
    # parser.add_argument("split")
    # parser.add_argument("ckpt_path")
    # parser.add_argument("layer", type=int)
    # parser.add_argument("nshard", type=int)
    # parser.add_argument("rank", type=int)
    # parser.add_argument("feat_dir")
    # parser.add_argument("--max_chunk", type=int, default=1600000)
    # args = parser.parse_args()
    # logger.info(args)

    # main(**vars(args))

    root = "data/speakers"
    split = "train"
    nshard = 1
    rank = 0
    feat_dir_base = "content_feats"

    ## Wav, No LA, No CL
    try:
        logger.info("Processing model type: Wav, No LA, No CL")
        model_name = "wav_nola_no_cl"
        resynthesizer = ResynthesisModule.load_from_checkpoint(
            "ckpts/resynthesis_la7.ckpt", 
            encoder_path=None, 
            map_location='cuda'
        )
        resynthesizer.eval()

        feat_dir = os.path.join(feat_dir_base, model_name)
        os.makedirs(feat_dir, exist_ok=True)
        main(root, split, resynthesizer, nshard, rank, feat_dir)
    except Exception as e:
        logger.error(f"Error processing model type: Wav, No LA, No CL - {e}")

