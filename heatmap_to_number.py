import numpy as np

import os
import glob


if __name__ == "__main__":
    path = os.getenv("GT_DIR") + "*.csv"

    counts = []
    for f in glob.glob(path):
        data = np.genfromtxt(f, delimiter=',')
        count = np.sum(data)
        counts.append(count)

    np.savetxt("counts.csv", counts, delimiter=",", fmt="%0.f")
