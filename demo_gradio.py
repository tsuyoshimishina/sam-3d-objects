# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import tempfile
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

import gradio as gr

# Load model once at startup
print("Loading model...")
config_path = "checkpoints/hf/pipeline.yaml"
inference = Inference(config_path, compile=False)
print("Model loaded successfully!")


def run_inference(image_file, mask_file, output_type, seed):
    """Run SAM 3D Objects inference on uploaded image and mask."""
    if image_file is None:
        raise gr.Error("Please upload an image file")
    if mask_file is None:
        raise gr.Error("Please upload a mask image file")

    # gr.File returns a file path string
    image_path = image_file if isinstance(image_file, str) else image_file.name
    mask_path = mask_file if isinstance(mask_file, str) else mask_file.name

    # Load image and mask
    image = load_image(image_path)
    mask = load_mask(mask_path)

    # Run inference
    seed_int = int(seed) if seed is not None else 42
    output = inference(image, mask, seed=seed_int)

    # Export output
    if output_type == "Gaussian Splat (.ply)":
        output_path = os.path.join(tempfile.gettempdir(), "output.ply")
        output["gs"].save_ply(output_path)
    else:
        glb = output["glb"]
        if glb is None:
            raise gr.Error("Mesh output is not available")
        output_path = os.path.join(tempfile.gettempdir(), "output.glb")
        glb.export(output_path)

    return output_path


def main():
    with gr.Blocks(title="SAM 3D Objects Demo") as demo:
        gr.Markdown("# SAM 3D Objects Demo")
        gr.Markdown("Upload an image and its mask to reconstruct 3D geometry.")

        with gr.Row():
            image_input = gr.File(label="Input Image", file_types=["image"])
            mask_input = gr.File(label="Mask Image", file_types=["image"])

        with gr.Row():
            output_type = gr.Radio(
                choices=["Gaussian Splat (.ply)", "Mesh (.glb)"],
                value="Gaussian Splat (.ply)",
                label="Output Type",
            )
            seed_input = gr.Number(value=42, label="Seed", precision=0)

        run_btn = gr.Button("Run Inference", variant="primary")
        output_file = gr.File(label="Output File")

        run_btn.click(
            fn=run_inference,
            inputs=[image_input, mask_input, output_type, seed_input],
            outputs=output_file,
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
