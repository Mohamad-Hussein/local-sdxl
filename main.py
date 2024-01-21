import os
import time
from datetime import datetime
from PIL import Image

from diffusers import DiffusionPipeline

import torch

import streamlit as st

######################################################
# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

# prompt = "A majestic lion jumping from a big stone at night"
######################################################

# Create a cache directory to store the models
cache_dir = "model/"
os.makedirs(cache_dir, exist_ok=True)


def generate_image(prompt: str, neg_prompt: str = None):
    """Generate an image from the prompt using the Stable Diffusion model
    It uses the base model, then removes the weights from the GPU and loads the refiner model
    to complete the image generation.

    Args:
        prompt (str): The prompt for the image
        neg_prompt (str, optional): The negative prompt for the image. Defaults to None.

    Returns:
        image: The generated image
        time_taken: Time taken to generate the image

    """

    torch.cuda.empty_cache()

    # Start time
    t0 = time.time()

    ## Base step
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_dir,
    )

    base.enable_model_cpu_offload()

    image = base(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    torch.cuda.empty_cache()

    ## Refiner step
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    refiner.enable_model_cpu_offload()

    images = refiner(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images

    # Extract the image from the list
    image = images[0]

    # Clearing everything
    torch.cuda.empty_cache()
    del base, refiner, images

    # Finish time
    t1 = time.time()

    return image, t1 - t0


def save_image_to_folder(image, folder):
    """Save the image to the specified folder"""
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
    image.save(f"{folder}/{filename}")


def main():
    # Set title
    st.title("Stable Diffusion Image Generation")

    # Text area for user prompt and negative prompt
    prompt = st.text_area("Enter a prompt for the image:")
    neg_prompt = st.text_area("Enter a negative prompt for the image:")

    # Button to generate the image
    if st.button("Generate Image"):
        if prompt:
            # Generate the image
            generated_image, time_taken = generate_image(prompt, neg_prompt)

            # Show image and time taken
            st.image(
                generated_image,
                caption=f"{int(time_taken)} seconds taken",
                use_column_width=True,
            )

            # Save the image to the "images" folder
            save_image_to_folder(generated_image, "images")


if __name__ == "__main__":
    main()
