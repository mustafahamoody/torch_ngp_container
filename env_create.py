import os
import subprocess
import shutil
import argparse

# ------------GLOBAL NERF SETUP------------
nerf_model = 'torch-ngp' # Default NeRF model to use. Other Option: 'd-nerf' -- Use d-nerf if video is dynamic.

bound = "2.0" # Default (Axis-ALigned) Bounding Box scale
scale = "0.33" # Default scale
dt_gamma = "0.02" # Default dt_gamma
# bg_radius = "32"

# Frame extraction (per second) for video
video_fps = "10" 

# ------------------------------------------


parser = argparse.ArgumentParser(description="Skip environment setup and directly generate NeRF model of the environment-- Used to adjust parameter when data is already setup")

parser.add_argument('--run', action='store_true', help="Run entire env_creation process")
parser.add_argument('--train', action='store_true', help="Skip to generating NeRF environment")
parser.add_argument('--view', action='store_true', help="View the generated NeRF model")
# parser.add_argument('--dynamic', action='store_true', help="Create dynamic NeRF model")

args = parser.parse_args()

def data_setup():
    # Training Info
    print("------------INFO: Environment Training Works Best With a 1+ Min Video OR a 50+ Images Folder ------------\n")

    # Get user input
    env_name = input("Enter Environment Name: ").strip()
    input_type = input("Image or Video: ").strip().lower()
    content_path = input("Enter Path to Video: ") if input_type == "video" else input("Enter Path to folder with images: ")

    # data folder
    data_folder = os.path.join(os.getcwd(), "data")
    os.makedirs("data", exist_ok=True)
    
    #Environment Folders Setup
    env_folder = os.path.join(os.getcwd(), data_folder, env_name)
    os.makedirs(env_folder, exist_ok=True)

    
    if input_type == "video":
        # Copy all video from the specified path to the environment folder
        shutil.copy(content_path, env_folder)
        print(f"Video copied from {content_path} to {env_folder} \n")
        video_path = os.path.join(env_folder, os.path.basename(content_path)) #Get the video path in the environment folder

    else:
        # Create images folder in the environment folder
        images_folder = os.path.join(env_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        # Copy all files from the specified folder path to the images folder
        for filename in os.listdir(content_path):
            source = os.path.join(content_path, filename)
            destination = os.path.join(images_folder, filename)
            if os.path.isfile(source):
                shutil.copy(source, destination)

        print(f"All images copied from {content_path} to {images_folder} \n")

    #Process Input Content
    colmap2nerf_path = os.path.join(os.getcwd(), "torch_ngp", "scripts", "colmap2nerf.py")
    
    if input_type == "video":
        # Extract frames from video using ffmpeg and then run colmap
        print("\n\n ---------------------------When Propted, Enter y to Run Program: --------------------------- \n\n")
        
        if nerf_model == "d-nerf":  #For dynamic scenes
            subprocess.run([
                "python", colmap2nerf_path, 
                "--video", video_path, 
                "--video_fps", video_fps, 
                "--hold", "0",
                "--run_colmap",
                "--dynamic"
            ])
        else:
            subprocess.run([
                "python", colmap2nerf_path, 
                "--video", video_path, 
                "--video_fps", video_fps, 
                "--hold", "0",
                "--run_colmap"
            ])

    else:
        # Run colmap on the images folder
        subprocess.run([
            "python", colmap2nerf_path,
            "--images", images_folder,
            "--hold", "0",
            "--run_colmap"
        ])

    print("Environment Folder Setup Complete \n")
    
    return env_name


def env_create(env_name):

    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"

    # Generate NeRF environment
    if  nerf_model == "d-nerf":
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_dnerf.py")
    else:
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")

    print("Generating NeRF Model of Environment \n")

    #May  need to set and try = different scale & bound & dt_gamma to make the object correctly located in the bounding box and render fluently.
    subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf, "-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma])
    
    print("NeRF Model Generated \n")
    
    print(f"NeRF Model of environment saved at {env_nerf} \n")
    
    return env_nerf

def env_view(env_name):

    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"
   
    if nerf_model == "d-nerf":
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_dnerf.py")
    else:
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")
    
    
    subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf,"-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma, "--gui", "--test"])

# Run the script based on the command line arguments
if args.train:
    env_name = input("Enter Environment Name (from setup): ").strip()
    env_create(env_name)

elif args.view:
    env_name = input("Enter Environment Name (from setup): ").strip()
    env_view(env_name)

elif args.run:
    env_name = data_setup()
    env_create(env_name)
    env_view(env_name)

else:
    env_name = data_setup()
    env_create(env_name)
    env_view(env_name)