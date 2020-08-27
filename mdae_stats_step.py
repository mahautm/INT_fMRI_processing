import sys
from mdae_stats import get_model_stats

if __name__ == "__main__":
    data_orig = sys.argv[1]  # {"ABIDE", "interTVA"}
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    dim = sys.argv[3]
    fold = int(sys.argv[4])  # int belonging to [1,10]
    get_model_stats(data_orig, data_type, dim, fold)

