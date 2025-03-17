from models import *

def main():
    dataset_name = "data"
    # dataset_name = "square-simple"
    arch = MLPArchitecture(3, [10, 10], 1)
    model = MLP(architecture=arch, dataset_name=dataset_name)
    model.forward()

if __name__ == "__main__":
    main()