import os
import subprocess
import shutil
import argparse

# ------------GLOBAL NERF SETUP------------
bound = "4.0" # Default (Axis-ALigned) Bounding Box scale
scale = "0.5" # Default scale
dt_gamma = "0.0" # Default dt_gamma
density_thresh = "10.0" # Default density threshold
iters = "40000" # Default number of iterations

# --------------DATA SETUP-------------
input_type = "image"

content_path = "data/my_scene_2"

video_fps = "10" # Frame extraction (per second) for video

# --------------ENVIRONMENT SCALING------------
# point1 = [0.3, 0.0, 0.0] # x, y, z coordinates of the first point
# point2 = [0.7, 0.0, 0.0] # x, y, z coordinates of the second point
real_world_distance = 5 # Real-world distance between the two points 

# ------------------------------------------


parser = argparse.ArgumentParser(description="Skip environment setup and directly generate NeRF model of the environment-- Used to adjust parameter when data is already setup")

parser.add_argument('--run', action='store_true', help="Run entire env_creation process")
parser.add_argument('--scale', action='store_true', help="Scale the environment to match real world size")
parser.add_argument('--train', action='store_true', help="Skip to generating NeRF environment")
parser.add_argument('--view', action='store_true', help="View the generated NeRF model")
parser.add_argument('--no_gui', action='store_true', help="Do not open GUI for colmap")
args = parser.parse_args()

def data_setup():
    """Setup the environment data"""
    print("------------INFO: Environment Training Works Best With a 1+ Min Video OR a 100+ Images Folder ------------\n")

    # Get user input
    env_name = input("Enter Environment Name: ").strip()
    print('')

    if input_type not in ["image", "video"]:
        print("Invalid input type. Please enter 'image' or 'video'.")
        exit(1)

    if not os.path.exists(content_path):
        print("Invalid content path. Please enter a valid path.")
        exit(1)

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


def scale_env(env_name):
    """Scale the environment"""
    env_folder = os.path.join(os.getcwd(), "data", env_name)

    # File Paths to open Colmap Point Cloud in GUI
    colmap_project_path = os.path.join(env_folder, "colmap_sparse", "0")
    colmap_db_path = os.path.join(env_folder, "colmap.db")
    image_path = os.path.join(env_folder, "images")

    # Run colmap to determine diffrence between 2 points
    if not args.no_gui:
        subprocess.run([
            "colmap", "gui",
            "--import_path", colmap_project_path,
            "--database_path", colmap_db_path,
            "--image_path", image_path
        ])


    # Get 2 points from user
    while True:
        point1 = input("Enter the first point: x, y, z: ").strip()
        point1 = [float(x) for x in point1.split(',')]
        point2 = input("Enter the second point: x, y, z: ").strip()
        point2 = [float(x) for x in point2.split(',')]
        if point1 and point2:
            break
        else:
            print("Invalid points. Please enter 2 points in x, y, z format.")

    scale_env_path = os.path.join(os.getcwd(), "scale_env.py")
    json_path = os.path.join(env_folder, "transforms.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(env_folder, "transforms_train.json")

    subprocess.run([
        "python", scale_env_path,
        "--json_path", json_path,
        "--real_distance", str(real_world_distance),
        "--point1", str(point1[0]), str(point1[1]), str(point1[2]),
        "--point2", str(point2[0]), str(point2[1]), str(point2[2])
    ])

def env_create(env_name):
    """Generate NeRF environment"""
    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"

    # Generate NeRF environment
    model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")

    print("Generating NeRF Model of Environment \n")

    try:
        #Train NeRF model on the environment
        subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf, "-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma, "--density_thresh", density_thresh, "--iters", iters])
    
    except Exception as e:
        print(f"Error generating NeRF model: {e}")
    
    print(f"--------------------------------NeRF Model Generration Complete: Model of environment saved at {env_nerf}--------------------------------")
    
    return env_nerf


def env_view(env_name):
    """View the generated NeRF model of the environment"""
    env_folder = os.path.join(os.getcwd(), "data", env_name)
    env_nerf = os.path.join(os.getcwd(), "data", env_name) + "_nerf"
   
    model_path = os.path.join(os.getcwd(), "torch_ngp", "main_nerf.py")
    
    
    subprocess.run(["python", model_path, env_folder, "--workspace", env_nerf,"-O", "--error_map","--bound", bound, "--scale", scale, "--dt_gamma", dt_gamma, "--density_thresh", density_thresh, "--gui", "--test"])


# Run the script based on the command line arguments
if args.train:
    env_name = input("Enter Environment Name (from setup): ").strip()
    env_create(env_name)

elif args.scale:
    env_name = input("Enter Environment Name (from setup): ").strip()
    scale_env(env_name)
    env_create(env_name)

elif args.view:
    env_name = input("Enter Environment Name (from setup): ").strip()
    env_view(env_name)

else:
    env_name = data_setup()
    scale_env(env_name)
    env_create(env_name)