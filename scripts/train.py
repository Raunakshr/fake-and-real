"""Training script for fake news detection.
TODO: Implement full training pipeline.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train models for fake news detection")
    parser.add_argument('--model', choices=['classical', 'deep'], default='classical')
    args = parser.parse_args()
    # TODO: Add training logic
    print(f"Training {args.model} model - TODO")


if __name__ == '__main__':
    main()
