# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

def main():
    parser = argparse.ArgumentParser(description="SAM 3D Objects inference demo")
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["gs", "mesh"],
        default="gs",
        help="Output type: 'gs' for Gaussian Splat (.ply), 'mesh' for Mesh (.glb)",
    )
    args = parser.parse_args()

    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    )
    mask = load_single_mask(
        "notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14
    )

    # run model
    output = inference(image, mask, seed=42)

    # export output
    if args.output_type == "gs":
        output["gs"].save_ply("output.ply")
        print("Gaussian Splat saved to output.ply")
    else:
        glb = output["glb"]
        if glb is None:
            print("Error: Mesh output is not available")
            sys.exit(1)
        glb.export("output.glb")
        print("Mesh saved to output.glb")

if __name__ == "__main__":
    main()
