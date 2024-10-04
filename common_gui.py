import os
import re
import gradio as gr
import sys
import json
import math
import shutil
import toml
from easygui import ynbox
from typing import Optional

# Set up logging
def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()

log = setup_logging()

folder_symbol = "\U0001f4c2"  # Folder icon
refresh_symbol = "\U0001f504"  # Refresh icon
save_style_symbol = "\U0001f4be"  # Save icon
document_symbol = "\U0001F4C4"  # Document icon

scriptdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if os.name == "nt":
    scriptdir = scriptdir.replace("\\", "/")

# Insert sd-scripts path into PYTHONPATH
sys.path.insert(0, os.path.join(scriptdir, "sd-scripts"))

# Model presets for validation
V2_BASE_MODELS = ["stabilityai/stable-diffusion-2-1-base", "stabilityai/stable-diffusion-2-base"]
V_PARAMETERIZATION_MODELS = ["stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-2"]
V1_MODELS = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]
SDXL_MODELS = ["stabilityai/stable-diffusion-xl-base-1.0"]

ALL_PRESET_MODELS = V2_BASE_MODELS + V_PARAMETERIZATION_MODELS + V1_MODELS + SDXL_MODELS

def check_if_model_exist(output_name: str, output_dir: str, save_model_as: str, headless: bool = False) -> bool:
    if headless:
        log.info("Headless mode, skipping verification if model already exists...")
        return False

    ckpt_file = os.path.join(output_dir, output_name + "." + save_model_as)
    if os.path.isfile(ckpt_file):
        log.info(f"A model file {ckpt_file} already exists.")
        return not ynbox(f"A model with the same file name exists. Overwrite?", "Overwrite")

    return False

def get_file_path(filename: str, directory: str) -> str:
    return os.path.join(directory, filename)

def validate_folder_path(folder: str) -> str:
    if os.path.isdir(folder):
        return folder
    else:
        raise FileNotFoundError(f"The folder {folder} does not exist.")

def validate_file_path(filename: str, directory: str) -> str:
    full_path = get_file_path(filename, directory)
    if os.path.isfile(full_path):
        return full_path
    else:
        raise FileNotFoundError(f"The file {filename} does not exist in {directory}.")

def validate_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    else:
        raise FileNotFoundError(f"The model path {model_path} does not exist.")

def get_saveasfile_path(output_dir: str, output_name: str, save_model_as: str) -> str:
    return os.path.join(output_dir, f"{output_name}.{save_model_as}")

def calculate_max_train_steps(total_steps: int, train_batch_size: int, gradient_accumulation_steps: int, epoch: int, reg_factor: int):
    return int(
        math.ceil(
            float(total_steps) / int(train_batch_size) / int(gradient_accumulation_steps) * int(epoch) * int(reg_factor)
        )
    )

def get_executable_path(executable_name: str = None) -> str:
    if executable_name:
        executable_path = shutil.which(executable_name)
        return executable_path if executable_path else ""  # Return empty if not found
    return ""  # Return empty if no executable name is provided

def output_message(msg: str = "", title: str = "", headless: bool = False) -> None:
    if headless:
        log.info(msg)
    else:
        ynbox(msg=msg, title=title)

def update_my_data(my_data):
    use_8bit_adam = my_data.get("use_8bit_adam", False)
    my_data.setdefault("optimizer", "AdamW8bit" if use_8bit_adam else "AdamW")

    model_list = my_data.get("model_list", [])
    pretrained_model_name_or_path = my_data.get("pretrained_model_name_or_path", "")
    if not model_list or pretrained_model_name_or_path not in ALL_PRESET_MODELS:
        my_data["model_list"] = "custom"

    # Update integer and float parameters
    for key in [
        "adaptive_noise_scale", "clip_skip", "epoch", "gradient_accumulation_steps",
        "keep_tokens", "lr_warmup", "max_data_loader_n_workers", "max_train_epochs",
        "save_every_n_epochs", "seed", "max_train_steps", "caption_dropout_every_n_epochs", "max_token_length"
    ]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                my_data[key] = int(0)

    for key in ["noise_offset", "learning_rate", "text_encoder_lr", "unet_lr", "lr_scheduler_power"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = float(value)
            except ValueError:
                my_data[key] = float(0.0)

    if my_data.get("LoRA_type", "Standard") == "LoCon":
        my_data["LoRA_type"] = "LyCORIS/LoCon"

    if "save_model_as" in my_data:
        if (
            my_data.get("LoRA_type") or my_data.get("num_vectors_per_token")
        ) and my_data.get("save_model_as") not in ["safetensors", "ckpt"]:
            my_data["save_model_as"] = "safetensors"

    xformers_value = my_data.get("xformers", None)
    if isinstance(xformers_value, bool):
        my_data["xformers"] = "xformers" if xformers_value else "none"

    if my_data.get("use_wandb") == "True":
        my_data["log_with"] = "wandb"

    my_data.pop("use_wandb", None)

    lora_network_weights = my_data.get("lora_network_weights")
    if lora_network_weights:
        my_data["network_weights"] = lora_network_weights

# Handle color augmentation changes
def color_aug_changed(color_aug):
    if color_aug:
        log.info('Disabling "Cache latent" because "Color augmentation" has been selected...')
        return gr.Checkbox(value=False, interactive=False)
    return gr.Checkbox(interactive=True)

# Additional functionalities to implement
def SaveConfigFile(config):
    pass  # Implement saving config to a file

def print_command_and_toml():
    pass  # Implement logic to print command and TOML configuration

def run_cmd_advanced_training():
    pass  # Implement logic for running advanced training commands

# Placeholder function for validate_args_setting
def validate_args_setting(args):
    pass  # Implement argument validation logic here
