"""
Created by: Liping Zhang
CUHK Lab of AI in Radiology (CLAIR)
Department of Imaging and Interventional Radiology
Faculty of Medicine
The Chinese University of Hong Kong (CUHK)
Email: lpzhang@link.cuhk.edu.hk
Copyright (c) CUHK 2023.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import argparse
from ktclair_test_examples.run_pretrained_ktclair_inference import run_inference

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', required=True, help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', required=True, help='output directory')
    parser.add_argument('--task', type=str, nargs='?', choices=("Cine","Mapping",), required=True, help='Cine/Mapping')
    parser.add_argument('--state_dict_file', type=str, nargs='?', default=None, help='Path to saved state_dict')

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    task = args.task
    state_dict_file = args.state_dict_file

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)
    print("task:", task)
    print("state_dict_file in:", state_dict_file)

    # TODO Load input file from input_dir and make your prediction,
    #  then output the predictions to output_dir
    run_inference(
        data_path=input_dir,
        output_path=output_dir,
        task=task,
        state_dict_file=state_dict_file,
    )
