"""Human Composition Data Generation Start!!!"""

import argparse
import datetime
import glob
import os
import shutil
import sys

import cv2
import lightning as L
import numpy as np
import rootutils
import torch
from einops import rearrange
from kornia.filters import gaussian_blur2d
from PIL import Image
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
sys.path.append("third_party/StyleGAN_Human")
sys.path.append("third_party/MaGGIe")
sys.path.append("third_party/PCTNet")
from src.models.components.human_composer import HumanComposer, HumanCompositionError

GENERATE_FROM_SCRATCH = False


def generate(
    dl3dv_dir: str = "/disk2/DL3DV-10K",
    outdir: str = "/disk2/DL3DV-10K_composed",
    total_gen_num: int = 100000,
    max_human_num_range: tuple[int, int] = (6, 46),
    human_min_depth_range: tuple[float, float] = (0.5, 5),
    human_max_depth_range: tuple[float, float] = (15, 20),
    device: str = "cuda:0",
):
    L.seed_everything(0)

    composer = (
        HumanComposer(
            max_human_num_range, human_min_depth_range, human_max_depth_range, verbose=False
        )
        .eval()
        .to(device)
    )

    ####

    date = datetime.date.today().strftime("%Y%m%d")

    currently_generated_image_num = 0
    for subfolder in range(1, 12):
        print(f"Subfolder: {subfolder}K start")

        scene_list = sorted(glob.glob(os.path.join(dl3dv_dir, f"{subfolder}K/*")))
        image_num_per_subfolder = (total_gen_num + 10) // 11
        image_num_per_scene = (image_num_per_subfolder + len(scene_list) - 1) // len(scene_list)

        currently_generated_image_per_subfolder = 0
        for scene_hash in tqdm(scene_list):

            image_list = sorted(glob.glob(os.path.join(scene_hash, "images_4/*.png")))
            interval = len(image_list) // image_num_per_scene

            if interval == 0:
                print(f"{scene_hash} has no images")
                continue

            savedir = os.path.join(outdir, f"{subfolder}K/{os.path.basename(scene_hash)}/images_4")
            os.makedirs(savedir, exist_ok=True)

            for imgpath in image_list[::interval]:
                frame_num = os.path.basename(imgpath).split(".")[0]
                composed_path = os.path.join(savedir, f"{frame_num}_composed_{date}.png")
                if not GENERATE_FROM_SCRATCH and os.path.isfile(composed_path):
                    print(f"Skipping {composed_path}...")
                    continue

                # load an image
                bg = cv2.imread(imgpath)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg_t = torch.tensor(bg, dtype=torch.float, device=device).permute(2, 0, 1) / 255
                img = torch.stack([bg_t], dim=0)
                _, _, height, width = img.shape

                # inference
                with torch.inference_mode():
                    try:
                        torch.cuda.empty_cache()
                        composed_rgba, composed_label, human_images = composer(img.to(device))

                        # apply gaussian blur (StyleGAN-Human is too crisp, so it doesn't fit the DL3DV-10K scenes)
                        apply_gaussian_blur = True
                        if apply_gaussian_blur:
                            rgb_ori, alpha = torch.split(composed_rgba, [3, 1], dim=1)
                            rgb_new = gaussian_blur2d(rgb_ori, (3, 3), (0.5, 0.5))
                            rgb_new = (1 - alpha) * rgb_ori + alpha * rgb_new
                            composed_rgba = torch.cat([rgb_new, alpha], dim=1)

                            batch, n, c, h, w = human_images.shape
                            rgb_ori, alpha = torch.split(human_images, [3, 1], dim=2)
                            rgb_new = rearrange(
                                gaussian_blur2d(
                                    rearrange(rgb_ori, "b n c h w -> (b n) c h w"),
                                    (3, 3),
                                    (0.5, 0.5),
                                ),
                                "(b n) c h w -> b n c h w",
                                b=batch,
                                n=n,
                            )
                            rgb_new = (1 - alpha) * rgb_ori + alpha * rgb_new
                            human_images = torch.cat([rgb_new, alpha], dim=2)

                        composed_label[composed_label < 0] = 255  # invalid value
                        composed_rgba = (
                            torch.clip(255 * composed_rgba.squeeze(0).permute(1, 2, 0), 0, 255)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
                        composed_label = (
                            torch.clip(composed_label.squeeze(), 0, 255)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
                        human_images = (
                            torch.clip(255 * human_images.squeeze(0).permute(0, 2, 3, 1), 0, 255)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
                    except HumanCompositionError as e:
                        torch.cuda.empty_cache()
                        print(f"{imgpath}: {e}")
                        continue

                # save memory
                torch.cuda.empty_cache()

                # save outputs
                area_per_label = np.bincount(composed_label.flatten())
                removed_labels = []
                if area_per_label.shape != (256,):
                    print(
                        f"\n\n\n[ERROR] Maybe the background is completely invisible? Skipping {imgpath}...\n({area_per_label=})\n\n\n"
                    )
                    continue
                for idx, person_rgba in enumerate(human_images):
                    assert person_rgba.shape == (height, width, 4), f"{person_rgba.shape=}"
                    if area_per_label[idx] < 64:
                        removed_labels.append(idx)
                        continue
                    Image.fromarray(person_rgba).save(
                        composed_path.replace(".png", f"_person_{idx}.png")
                    )

                composed_label = np.where(
                    np.isin(composed_label, removed_labels), 255, composed_label
                )
                Image.fromarray(composed_rgba).save(composed_path)
                Image.fromarray(composed_label).save(
                    composed_path.replace(".png", "_label.png")
                )  # DON'T USE JPEG!!!!!!

                # increment the counter
                currently_generated_image_per_subfolder += 1
                currently_generated_image_num += 1

            if currently_generated_image_per_subfolder >= image_num_per_subfolder:
                break

        if currently_generated_image_num >= total_gen_num:
            break

    print(f"Done ({currently_generated_image_num=})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dl3dv_dir",
        type=str,
        default="/disk1/ryotaro/data/DL3DV-10K",
        help="DL3DV-10K root path",
    )
    parser.add_argument(
        "--outdir", type=str, default="/disk1/ryotaro/data/DL3DV-10K_composed", help="Save path"
    )
    parser.add_argument(
        "--total_gen_num", type=int, default=100000, help="Number of generating images"
    )
    parser.add_argument(
        "--max_human_num_range",
        type=int,
        nargs="+",
        default=(6, 48),
        help="Maximum number of people in a scene",
    )
    parser.add_argument(
        "--human_min_depth_range",
        type=float,
        nargs="+",
        default=(0.5, 5),
        help="human-placed minimum depth range",
    )
    parser.add_argument(
        "--human_max_depth_range",
        type=float,
        nargs="+",
        default=(15, 20),
        help="human-placed minimum depth range",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # save args
    cmd = "python scripts/human_composition_data_gen.py "
    for key, val in vars(args).items():
        print(f"{key}: {val}")
        cmd += f"--{key} {val} "

    # create save dir
    if os.path.isdir(args.outdir):
        while True:
            inp = input(
                f"\n{args.outdir=} already exists. Do you Add data in it? [y(yes)/n(no)/d(delete)]: "
            )
            if inp == "y":
                break
            elif inp == "n":
                exit()
            elif inp == "d":
                GENERATE_FROM_SCRATCH = True
                shutil.rmtree(args.outdir)
                break
            else:
                print(f"Invalid input: {inp}")

    # save the composer file and this file
    os.makedirs(args.outdir, exist_ok=True)
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "human_composition_data_gen.py"), args.outdir
    )
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "../src/models/components/human_composer.py"),
        args.outdir,
    )
    with open(os.path.join(args.outdir, "cmd.txt"), "w") as f:
        f.write(cmd)

    # run
    generate(
        dl3dv_dir=args.dl3dv_dir,
        outdir=args.outdir,
        total_gen_num=args.total_gen_num,
        max_human_num_range=args.max_human_num_range,
        human_min_depth_range=args.human_min_depth_range,
        human_max_depth_range=args.human_max_depth_range,
        device=args.device,
    )
