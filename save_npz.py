import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="samples")
args = parser.parse_args()

from sample_ddp import create_npz_from_sample_folder
import os
#num_images = min( len(os.listdir(args.image_dir)), 50_000 )
#print(f"Total number of images that will be sampled: {num_images}")
create_npz_from_sample_folder(args.image_dir, 50_000)