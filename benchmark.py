import csv
import argparse
import os

# parse script arguments
parser = argparse.ArgumentParser(description='Run RawSplat training using configurations from CSV file')
parser.add_argument('-f', '--csv', type=str, help='Path to the CSV file')
parser.add_argument('-d', '--data', type=str, help='Path to dataset')
parser.add_argument('-o', '--output', type=str, help='Path to output directory that stores all benchmark results')
args = parser.parse_args()

# extract training parameters from CSV file
def get_param_vals_from_csv(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the first row as header
        cfgs = [] # list of dictionaries containing configurations for training

        # iterate through CSV rows
        for row in reader:
            run_cfg = {}
            for i, value in enumerate(row):
                key = header[i]
                run_cfg[key] = value
            cfgs.append(run_cfg)
    return cfgs

def format_train_args(cfg, data_path, output_path, idx):
    train_args = [] # string

    cfg_init = cfg.pop('init')
    cfg_img = cfg.pop('img')

    # source location (-s)
    src_path = data_path
    src_path += ('\colmap_rgb' if cfg_init == 'rgb' else '\\colmap_raw')
    train_args.append('-s')
    train_args.append(src_path)

    # image set location (-i)
    img_path = data_path
    img_path += ('\\8b-post-png' if cfg_img == 'rgb' else '\\raw')
    train_args.append('-i')
    train_args.append(img_path)

    # model output location (-m)
    model_path = output_path
    if cfg_img == 'rgb':
        model_path += '\\rgb'
    elif cfg_img == 'bayer':
        model_path += '\\bayer'
        train_args.append('--resolution')
        train_args.append('1')
    else:
        model_path += '\\raw'
    model_path += '\\' + cfg_init + '-init'
    model_path += '\\loss-' + cfg['loss_type']
    model_path += '\\run-' + str(idx)
    train_args.append('-m')
    train_args.append(model_path)


    # training hyperparams
    for key, value in cfg.items():
        train_args.append('--' + key)
        train_args.append(value)

    return train_args


if __name__ == '__main__':
    # parse csv config file
    csv_path = args.csv
    data_path = args.data
    output_path = args.output
    cfgs = get_param_vals_from_csv(csv_path)

    # start training
    for i, c in enumerate(cfgs):
        cmd = ['python', 'train.py']
        train_args = format_train_args(c, data_path, output_path, i)
        cmd.extend(train_args)
        str_cmd = ' '.join(cmd)
        os.system(str_cmd)