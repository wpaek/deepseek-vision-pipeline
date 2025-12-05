"""CLI entry point for the pipeline."""

import argparse
from pipeline import DeepSeekPipeline


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Vision Pipeline")
    parser.add_argument('--images-dir', required=True, help='Directory containing images')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--num-images', type=int, help='Max images to process')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    pipeline = DeepSeekPipeline(args.config)
    pipeline.run(args.images_dir, args.output_dir, args.num_images)


if __name__ == '__main__':
    main()
