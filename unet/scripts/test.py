import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", help = "tests", nargs='+', type=int, required=True)
args = parser.parse_args()

args.test = tuple(args.test)
print("Given arguments:")
print(args)