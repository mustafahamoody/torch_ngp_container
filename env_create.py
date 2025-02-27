import os
import subprocess
import shutil
import argparse

# ------------GLOBAL NERF SETUP------------
nerf_model = 'torch-ngp' # Default NeRF model to use. Other Option: 'd-nerf' -- Use d-nerf if video is dynamic.

bound = "2.0" # Default (Axis-ALigned) Bounding Box scale
scale = "0.5" # Default scale
dt_gamma = "0.0" # Default dt_gamma
density_thresh = "10.0" # Default density threshold
iters = "40000" # Default number of iterations

video_fps = "10" # Frame extraction (per second) for video


# --------------DATA SETUP--------------
input_type = "video"

content_path = "data/box.mp4"

# ------------------------------------------


parser = argparse.ArgumentParser(description="Skip environment setup and directly generate NeRF model of the environment-- Used to adjust parameter when data is already setup")

parser.add_argument('--run', action='store_true', help="Run entire env_creation process")
parser.add_argument('--train', action='store_true', help="Skip to generating NeRF environment")
parser.add_argument('--view', action='store_true', help="View the generated NeRF model")

args = parser.parse_args()

def data_setup():
    """Setup the environment data"""
    print("------------INFO: Environment Training Works Best With a 1+ Min Video OR a 100+ Images Folder ------------\n")

    # Get user input
    env_name = input("Enter Environment Name: ").strip()
    print('')

    while True:
        if input_type in ["image", "video"]:
            break
        else:
            print("Invalid input type. Please enter 'image' or 'video'.")
            print('')

    while True:
        if os.path.exists(content_path):
            break
        else:
            print("Invalid content path. Please enter a valid path.")
            print('')

    if input(f'WARNING: This process will create a new environment with the name \"{env_name}\" \n \
             or DELETE AND REPLACE ALL DATA if the environment already exists. Continue? (y/n): ') == 'n':
        return
    print('')
    
    # data folder
    data_folder = os.path.join(os.getcwd(), "data")
    os.makedirs("data", exist_ok=True)
    
    #Environment Folders Setup
    env_folder = os.path.join(os.getcwd(), data_folder, env_name)
    os.makedirs(env_folder, exist_ok=True)

    # Create images folder in the environment folder
    images_folder = os.path.join(env_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    if input_type == "video":
        # Copy all video from the specified path to the environment folder
        shutil.copy(content_path, env_folder)
        print(f"Video copied from {content_path} to {env_folder} \n")
        video_path = os.path.join(env_folder, os.path.basename(content_path)) #Get the video path in the environment folder
        print(video_path)

    else:
        # Copy all (image) files from the specified folder path to the images folder
        for filename in os.listdir(content_path):
            source = os.path.join(content_path, filename)
            destination = os.path.join(images_folder, filename)
            if os.path.isfile(source):
                shutil.copy(source, destination)

        print(f"All images copied from {content_path} to {images_folder} \n")

    # Run ffmpeg to extract frames from video
    if input_type == "video":
        subprocess.run([
            'ffmpeg', "-i", video_path, "-qscale:v", "1", "-qmin", "1", "-vf", f"fps={video_fps}", images_folder + "/%04d.jpg"
        ])

        print(f"Video frames extracted from {video_path} to {images_folder} \n")

    # Run remove_blurry to remove blurry images
    remove_blurry_path = os.path.join(os.getcwd(), "torch_ngp", "scripts", "remove_blurry.py")
    subprocess.run([
        "python", remove_blurry_path,
        "--data", str(images_folder),
        "--blur_threshold", "100"
    ])

    #Run Colmap on the images folder to extract camera poses
    colmap2nerf_path = os.path.join(os.getcwd(), "torch_ngp", "scripts", "colmap2nerf.py")
    
    try:
        subprocess.run([
            "python", colmap2nerf_path,
            "--images", images_folder,
            "--run_colmap",
            "--estimate_affine_shape"
        ])

        print("Environment Folder Setup Complete \n")

    except Exception as e:
        print(f"Error running colmap: {e}")
    
    return env_name


def env_create(env_name):
    """Generate NeRF environment"""
    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"

    # Generate NeRF environment
    if  nerf_model == "d-nerf":
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_dnerf.py")
    else:
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")

    print("Generating NeRF Model of Environment \n")

    #May  need to set and try = different scale & bound & dt_gamma to make the object correctly located in the bounding box and render fluently.
    subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf, "-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma, "--density_thresh", density_thresh, "--iters", iters])
    
    print("NeRF Model Generated \n")
    
    print(f"NeRF Model of environment saved at {env_nerf} \n")
    
    return env_nerf


def env_view(env_name):
    """View the generated NeRF model of the environment"""
    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"
   
    if nerf_model == "d-nerf":
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_dnerf.py")
    else:
        model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")
    
    
    subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf,"-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma, "--density_thresh", density_thresh, "--gui", "--test"])


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