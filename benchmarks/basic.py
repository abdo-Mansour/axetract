from axetract import AXEPipeline


def main():
    pipeline = AXEPipeline.from_config()

    result = pipeline.extract(
        input_data="https://en.wikipedia.org/wiki/Napoleonic_Wars",
        query="when did the wars start?"
    )

    print(result)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # profiler.dump_stats("benchmarks/profile.stats")
    # print("\nProfiling complete. Stats saved to 'benchmarks/profile.stats'")
    # print("Run 'snakeviz benchmarks/profile.stats' to view the profile.")
