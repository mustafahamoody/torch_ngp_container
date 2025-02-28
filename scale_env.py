import json
import numpy as np
import argparse
import os

def determine_scale(point1, point2, real_word_distance):
    delta = np.linalg.norm([np.array(point1) - np.array(point2)])
    scaling_factor = delta/real_word_distance
    print(f'------------------Scaling Environment by a factor of {scaling_factor}------------------')
    return scaling_factor

def scale_transforms(json_path, scaling_factor,  output_path='transforms_scaled.json'):
    # Load original transforms.json
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Apply scaling to transform_matrix for each frame
    for frame in data['frames']:
        transform_matrix = frame['transform_matrix']

        # scale the translation part of the matrix (last column, first 3 values)
        for i in range(3):
            transform_matrix[i][3] *= scaling_factor

        frame['transform_matrix'] = transform_matrix

    # Save modified transforms.json
    with open(output_path, 'w') as f:
        json.dump(data, indent=4)

    print(f'Scaled transforms saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, help='Path to the transforms.json file')
    parser.add_argument('--real_distance', type=float, required=True, help='Real-world distance between the two points')
    parser.add_argument('--point1', type=float, nargs=3, required=True, help='x, y, or z coordinates of the first point')
    parser.add_argument('--point2', type=float, nargs=3, required=True, help='x, y, or z coordinates of the second point')
    args = parser.parse_args()

    # Rename transforms.json to transforms_unscaled.json
    os.rename(args.json_path, args.json_path.replace('.json', '_unscaled.json'))
    original_json_path = args.json_path.replace('.json', '_unscaled.json')

    # Determine the scaling factor and scale the transforms accordingly
    scaling_factor = determine_scale(args.point1, args.point2, args.real_distance)
    scale_transforms(original_json_path, scaling_factor, args.json_path)