import argparse
import sys

from stable_baselines3 import PPO
from rich.console import Console
from rich.syntax import Syntax
from python_scripts.crazyflie_firmware_nn_compute.c_code_blocks import (
    headers_nn_compute,
    output_arrays,
    forward_pass_function
)


def get_policy_parameters_ids_with(extractor: str, parameter: str, model_parameters: dict) -> list:
    return list(filter(lambda k: extractor in k and parameter in k, model_parameters.keys()))


def get_policy_parameters_ids(model_parameters: dict) -> tuple:
    return (
        (
            *get_policy_parameters_ids_with("policy", "weight", model_parameters),
            *get_policy_parameters_ids_with("action", "weight", model_parameters)
        ),
        (
            *get_policy_parameters_ids_with("policy", "bias", model_parameters),
            *get_policy_parameters_ids_with("action", "bias", model_parameters)
        )
    )


def get_model_parameters(model_path: str) -> dict:
    model = PPO.load(model_path)
    return model.policy.state_dict()


def get_policy_structure(model_parameters: dict) -> tuple:
    ids = get_policy_parameters_ids_with("policy", "weight", model_parameters)
    ids.append(*get_policy_parameters_ids_with("action", "weight", model_parameters))

    return tuple((model_parameters[p].shape[0], model_parameters[p].shape[1]) for p in ids)


def string_policy_structure(structure: tuple) -> str:
    source = "static const int structure[3][2] = {"

    for layer in structure:
        source += f"{{{layer[0]}, {layer[1]}}},"

    source = source[:-1]
    source += "};"
    return source


def string_layer_weights(layers_ids: tuple, model_parameters: dict) -> str:
    source = ""
    structure = get_policy_structure(model_parameters)

    for i, layer_id in enumerate(layers_ids):
        source += f"static const float {layer_id.replace('.', '_')}[{structure[i][0]}][{structure[i][1]}] = {{"
        for row in model_parameters[layer_id]:
            source += f"{{{','.join(map(str, row.tolist()))}}},"
        source = source[:-1]
        source += "};\n"

    return source


def string_layer_bias(layers_ids: tuple, model_parameters: dict) -> str:
    source = ""
    structure = get_policy_structure(model_parameters)

    for i, layer_id in enumerate(layers_ids):
        source += f"static const float {layer_id.replace('.', '_')}[{structure[i][0]}] = {{"
        source += f"{','.join(map(str, model_parameters[layer_id].tolist()))}"
        source += "};\n"

    return source


def code_preview(code: str) -> None:
    console = Console()
    syntax = Syntax(code, "c", theme="monokai", line_numbers=True)
    console.print(syntax)


def generate_c_nn_compute(model_path: str, forward_pass_method: str, output_file: str, preview: bool = \
    False):
    c_source = ""
    try:
        model_params = get_model_parameters(model_path)
    except FileNotFoundError:
        sys.exit(f"[!] No model file in {model_path}.")

    layers_weights_ids, layer_bias_ids = get_policy_parameters_ids(model_params)

    c_source += headers_nn_compute[forward_pass_method]
    c_source += string_policy_structure(get_policy_structure(model_params))
    c_source += output_arrays
    c_source += string_layer_weights(layers_weights_ids, model_params)
    c_source += string_layer_bias(layer_bias_ids, model_params)
    c_source += forward_pass_function[forward_pass_method]

    if preview:
        code_preview(c_source)

    with open(output_file, "w") as f:
        f.write(c_source)


def get_execution_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate an MLP model C source code from a given Stable-Baselines3 trained model')

    parser.add_argument(
        'model',
        type=str,
        help="The path should point to a zip file containing the trained model"
    )
    parser.add_argument(
        '-f', '--forward-pass-method',
        type=str,
        default="for-based",
        choices=["for-based", "unrolled-for"],
        help="The method to compute the forward pass (for loop based or unrolled for)."
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default="./nn_compute.c",
        help="The path and filename of the output C source code."
    )
    parser.add_argument(
        '-p', '--code-preview',
        type=bool,
        help="Print with syntax highlighting the generated C code"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_execution_arguments()

    generate_c_nn_compute(args.model, args.forward_pass_method, args.output_file, args.code_preview)
