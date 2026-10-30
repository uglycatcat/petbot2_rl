"""Reserved entry for the independent rebuild. No training implementation yet."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.exit(2, 'Training is paused for a clean rebuild and is not implemented. '
                   'Read README.md and docs/rebuild_plan.md. No legacy fallback is available.\n')


if __name__ == '__main__':
    main()
