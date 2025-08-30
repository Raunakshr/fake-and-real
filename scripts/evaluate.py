"""Evaluation script for fake news detection models.
TODO: Implement evaluation routine.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models")
    parser.add_argument('--model-path', type=str, default='../models/classical_best.pkl')
    args = parser.parse_args()
    # TODO: Add evaluation logic
    print(f"Evaluating model at {args.model_path} - TODO")


if __name__ == '__main__':
    main()
