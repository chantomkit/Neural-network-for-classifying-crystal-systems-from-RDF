import numpy as np
from tqdm import tqdm

binmax = 1000
# distribution ratio for each crystal system
ratio = np.array([1, 1, 1, 1, 1, 1, 1])
filename = str(ratio).replace(" ", "").replace("[", "").replace("]", "")
rdf = f"td_fullrdf_mp"

samples = np.load(f"{rdf}/training_data_mp_{binmax}_pbc_big.npy", allow_pickle=True)
print(f"Batch size: {len(samples)}")

system = np.zeros(7)
spgrp = np.zeros(230, dtype=int)
for X, y, z in samples:
    system[np.argmax(y)] += 1
    spgrp[np.argmax(z)] += 1
print(f"Initial dataset: {system}\n{spgrp}")

# # Balancing datasets for each space group
# np.random.shuffle(samples)
# spgrp = np.zeros(230, dtype=int)
# for X, y, z in samples:
#     spgrp[np.argmax(z)] += 1
# print("Balancing datasets")
# for i in tqdm(range(len(spgrp))):
#     j = 0
#     nToDel = spgrp[i] - 100
#     while j < len(samples):
#         if nToDel > 0:
#             if int(np.argmax(samples[j][2])) == i:
#                 samples = np.delete(samples, [j][:], axis=0)
#                 j -= 1
#                 nToDel -= 1
#         j += 1

# Balancing datasets for each crystal system
# Number of samples per group (target number)
n_pergroup = 4100

np.random.shuffle(samples)
system = np.zeros(7, dtype=int)
for X, y, z in samples:
    system[np.argmax(y)] += 1
print("Balancing datasets")
for i in tqdm(range(len(system))):
    j = 0
    nToDel = system[i] - int(n_pergroup * ratio[i])
    while j < len(samples):
        if nToDel > 0:
            if int(np.argmax(samples[j][1])) == i:
                samples = np.delete(samples, [j][:], axis=0)
                j -= 1
                nToDel -= 1
        j += 1

# Displaying final distribution of dataset
system = np.zeros(7)
spgrp = np.zeros(230, dtype=int)
for X, y, z in samples:
    system[np.argmax(y)] += 1
    spgrp[np.argmax(z)] += 1
print(f"Final dataset: {system}\n{spgrp}")

np.random.shuffle(samples)
# Filename after packing and balancing
np.save(f"packed_td/{rdf}_{filename}_{binmax}.npy", samples)