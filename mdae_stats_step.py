import sys
from mdae_stats import get_model_stats

if __name__ == "__main__":
    data_orig = sys.argv[1]  # {"ABIDE", "interTVA"}
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    dim = sys.argv[3]
    nb_fold = int(sys.argv[4])  # int belonging to [1,10]
    print(dim)
    get_model_stats(data_orig, data_type, dim, nb_fold)

