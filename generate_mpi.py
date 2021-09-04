import numpy as np
import os
import tensorflow as tf
from single_view_mpi.libs import mpi
from single_view_mpi.libs import nets
from imageio import imwrite
from functools import lru_cache
import glob


@lru_cache(maxsize=1)
def create_model():
    input1 = tf.keras.Input(shape=(None, None, 3))
    output = nets.mpi_from_image(input1)

    model = tf.keras.Model(inputs=input1, outputs=output)
    print("Model created.")
    # Our full model, trained on RealEstate10K.
    model.load_weights(
        "single_view_mpi_full_keras/single_view_mpi_keras_weights"
    ).expect_partial()
    return model


def generate_mpi(input_path):

    model = create_model()
    input_rgb = tf.image.decode_image(tf.io.read_file(input_path), dtype=tf.float32)

    # input_rgb = tf.image.resize(input_rgb, (output_height,output_width), method='area')

    height, width = input_rgb.shape[:2]
    padding = width // 2

    left = input_rgb[:, 0:padding]
    right = input_rgb[:, width - padding : width]

    input_rgb = np.concatenate((right, input_rgb, left), axis=1)

    # Generate MPI
    layers = model(input_rgb[tf.newaxis])[0]
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    # Layers is now a tensor of shape [L, H, W, 4].
    # This represents an MPI with L layers, each of height H and width W, and
    # each with an RGB+Alpha 4-channel image.

    # Depths is a tensor of shape [L] which gives the depths of the L layers.

    # Display computed disparity
    disparity = mpi.disparity_from_layers(layers, depths)

    layers = [i[:, padding:-padding] for i in layers]
    disparity_map = disparity[..., 0][:, padding:-padding]

    return disparity_map, layers

    my_output_path = os.path.join(output_path, os.path.basename(path).split(".")[0])
    os.makedirs(my_output_path, exist_ok=True)

    imwrite(f"{my_output_path}/depth_map.png", disparity[..., 0][:, padding:-padding])

    os.makedirs(f"{my_output_path}/layers", exist_ok=True)

    for i in range(32):
        print(f"{my_output_path}/layers/layer_{i}.png")
        imwrite(
            f"{my_output_path}/layers/layer_{i}.png",
            layers[i].numpy()[:, padding:-padding],
        )


if __name__ == "__main__":
    from configargparse import ArgumentParser
    import glob
    import os

    parser = ArgumentParser(description="Generate Depth Maps")

    parser.add_argument("--input", required=True, help="input image or directory")
    parser.add_argument("--width", required=True, type=int, help="output image width")
    parser.add_argument("--height", required=True, type=int, help="output image height")
    parser.add_argument(
        "--output", "-o", required=True, help="directory for cylindrical output"
    )

    args = parser.parse_args()

    # verify/create the output directory
    os.makedirs(args.output, exist_ok=True)

    generate_mpi(
        args.input, args.output, output_width=args.width, output_height=args.height
    )
