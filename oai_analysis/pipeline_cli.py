# Usage
# python oai_analysis/pipeline_cli.py oai_analysis/data/test_data/colab_case/image_preprocessed.nii.gz OAI_results

import argparse
from pipeline import analysis_pipeline


def main():
    parser = argparse.ArgumentParser(description='OAI Analysis CLI')
    parser.add_argument('input_path', type=str, help='Path to image file or directory containing DICOM series')
    parser.add_argument('output_dir', type=str, help='Directory to make output files')
    parser.add_argument(
        '--no_intermediates',
        action='store_true',
        help='Do not write files representing intermediate steps to the output directory',
    )

    args = parser.parse_args()

    analysis_pipeline(args.input_path, args.output_dir, not args.no_intermediates)


if __name__ == '__main__':
    main()
