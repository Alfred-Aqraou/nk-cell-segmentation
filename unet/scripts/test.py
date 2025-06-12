from typing import Tuple
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--test", help = 'tests', required=True, type=str)
args = parser.parse_args()
print("Given arguments:")
print(args)

args.test = str(args.test).strip().lower() in ("true", "1", "yes", "y", "t")
print(args.test)