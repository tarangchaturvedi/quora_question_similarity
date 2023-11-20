import warnings
warnings.filterwarnings('ignore')
import argparse
from train_model import train_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train_model(args.train_file)
    elif args.mode == "test":
        if args.prediction_file is not None:
            test_model(args.test_file, args.model, output_csv_file=args.prediction_file)
        else:
            test_model(args.test_file, args.model, output_csv_file='predictions.csv')
    else:
        print("wrong mode")
