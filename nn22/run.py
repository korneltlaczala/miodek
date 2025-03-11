from models import *

def main():
    dataset_name = "square-simple"
    arch = MLPArchitecture(1, [10, 10], 1)
    model = MLP(architecture=arch, dataset_name=dataset_name)

if __name__ == "__main__":
    main()