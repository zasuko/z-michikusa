import os
import re
import gradio as gr
import sys
import json
import math
import shutil
import toml
import requests
from easygui import ynbox
from typing import Optional

# Set up logging
def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()

log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

scriptdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if os.name == "nt":
    scriptdir = scriptdir.replace("\\", "/")

# insert sd-scripts path into PYTHONPATH
sys.path.insert(0, os.path.join(scriptdir, "sd-scripts"))

# GitHubからcommon_gui.pyをダウンロードする関数
def download_common_gui(url: str, save_path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        log.info("common_gui.py downloaded successfully.")
    else:
        log.error(f"Failed to download common_gui.py, status code: {response.status_code}")

# common_gui.pyのパス
common_gui_url = "https://raw.githubusercontent.com/zasuko/z-michikusa/main/common_gui.py"
common_gui_path = os.path.join(scriptdir, "common_gui.py")

# common_gui.pyをダウンロードして上書き
download_common_gui(common_gui_url, common_gui_path)

# Model presets for validation
V2_BASE_MODELS = ["stabilityai/stable-diffusion-2-1-base", "stabilityai/stable-diffusion-2-base"]
V_PARAMETERIZATION_MODELS = ["stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-2"]
V1_MODELS = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]
SDXL_MODELS = ["stabilityai/stable-diffusion-xl-base-1.0"]

ALL_PRESET_MODELS = V2_BASE_MODELS + V_PARAMETERIZATION_MODELS + V1_MODELS + SDXL_MODELS

def check_if_model_exist(output_name: str, output_dir: str, save_model_as: str, headless: bool = False) -> bool:
    """
    Checks if a model with the same name already exists and prompts the user to overwrite it if it does.
    """
    if headless:
        log.info("Headless mode, skipping verification if model already exists...")
        return False

    if save_model_as in ["diffusers", "diffusers_safetensors"]:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            log.info(f"A model folder {ckpt_folder} already exists.")
            return not ynbox(f"A model with the same folder name exists. Overwrite?", "Overwrite")
    else:
        ckpt_file = os.path.join(output_dir, output_name + "." + save_model_as)
        if os.path.isfile(ckpt_file):
            log.info(f"A model file {ckpt_file} already exists.")
            return not ynbox(f"A model with the same file name exists. Overwrite?", "Overwrite")

    return False

def get_file_path(filename: str, directory: str) -> str:
    """
    Returns the full path to the file in the specified directory.
    """
    return os.path.join(directory, filename)

def get_saveasfile_path(filename: str, output_dir: str) -> str:
    """
    Returns the full path to save the specified filename in the output directory.
    """
    return os.path.join(output_dir, filename)

def calculate_max_train_steps(total_steps: int, train_batch_size: int, gradient_accumulation_steps: int, epoch: int, reg_factor: int):
    return int(
        math.ceil(
            float(total_steps) / int(train_batch_size) / int(gradient_accumulation_steps) * int(epoch) * int(reg_factor)
        )
    )

def get_executable_path(executable_name: str = None) -> str:
    """
    Retrieve and sanitize the path to an executable in the system's PATH.
    """
    if executable_name:
        executable_path = shutil.which(executable_name)
        if executable_path:
            return executable_path
        else:
            return ""  # Return empty string if the executable is not found
    else:
        return ""  # Return empty string if no executable name is provided

def output_message(msg: str = "", title: str = "", headless: bool = False) -> None:
    """
    Outputs a message to the user, either in a message box or in the log.
    """
    if headless:
        log.info(msg)
    else:
        ynbox(msg=msg, title=title)

def update_my_data(my_data):
    """
    Processes and updates the data, converts strings to integers or floats where necessary.
    """
    use_8bit_adam = my_data.get("use_8bit_adam", False)
    my_data.setdefault("optimizer", "AdamW8bit" if use_8bit_adam else "AdamW")

    model_list = my_data.get("model_list", [])
    pretrained_model_name_or_path = my_data.get("pretrained_model_name_or_path", "")
    if not model_list or pretrained_model_name_or_path not in ALL_PRESET_MODELS:
        my_data["model_list"] = "custom"

    for key in [
        "adaptive_noise_scale",
        "clip_skip",
        "epoch",
        "gradient_accumulation_steps",
        "keep_tokens",
        "lr_warmup",
        "max_data_loader_n_workers",
        "max_train_epochs",
        "save_every_n_epochs",
        "seed",
    ]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                my_data[key] = int(0)

    for key in ["lr_scheduler_num_cycles"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                my_data[key] = int(1)

    for key in ["max_train_steps", "caption_dropout_every_n_epochs"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                my_data[key] = int(0)

    for key in ["max_token_length"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                my_data[key] = int(75)

    for key in ["noise_offset", "learning_rate", "text_encoder_lr", "unet_lr"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = float(value)
            except ValueError:
                my_data[key] = float(0.0)

    for key in ["lr_scheduler_power"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = float(value)
            except ValueError:
                my_data[key] = float(1.0)

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
        my_data.pop("lora_network_weights", None)

    return my_data

# Add the missing function for color_aug_changed
def color_aug_changed(color_aug):
    """
    Handles the change in color augmentation checkbox.
    Disables the 'cache latent' option if color augmentation is enabled.
    """
    if color_aug:
        log.info('Disabling "Cache latent" because "Color augmentation" has been selected...')
        return gr.Checkbox(value=False, interactive=False)
    else:
        return gr.Checkbox(interactive=True)

# Add the print_command_and_toml function
def print_command_and_toml(command: str, toml_dict: dict) -> None:
    """
    Prints the command and the corresponding TOML configuration.
    """
    log.info(f"Command: {command}")
    log.info("TOML Configuration:")
    for key, value in toml_dict.items():
        log.info(f"{key}: {value}")
