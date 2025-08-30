"""Command line interface for quick predictions.
TODO: Implement inference using saved pipeline.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Fake news detector CLI")
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    args = parser.parse_args()
    # TODO: Load pipeline and predict
    print(f"Input: {args.text}\nPrediction: TODO")


if __name__ == '__main__':
    main()
