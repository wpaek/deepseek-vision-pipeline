"""Example usage."""

from pipeline import DeepSeekPipeline


def main():
    pipeline = DeepSeekPipeline("config.yaml")
    dataset = pipeline.run("./data/images", "./output", num_images=10)
    
    for i, item in enumerate(dataset[:3], 1):
        print(f"\n{i}. {item['image']}")
        print(f"Bin: {item['bin']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print(f"R: {item['reasoning'][:100]}...")


if __name__ == '__main__':
    main()
