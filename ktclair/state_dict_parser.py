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
from utils import save_state_dict_from_checkpoint

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, nargs='?', default=None, required=True, help='checkpoint_path')
    parser.add_argument('--state_dict_path', type=str, nargs='?', default=None, required=True, help='state_dict_path')
    parser.add_argument('--net_startswith', type=str, nargs='?', default='model.', help='net startswith')
 
    args = parser.parse_args()

    print("Checkpoint store in:", args.checkpoint_path)
    print("State_dict store in:", args.state_dict_path)
    print("net startswith key:", args.net_startswith)

    # TODO Load input file from input_dir and make your prediction,
    #  then output the predictions to output_dir
    save_state_dict_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        state_dict_path=args.state_dict_path,
        net_startswith=args.net_startswith,
    )

    print("Done")
