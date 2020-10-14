import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 30
binmax = 1000

types = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
ratio = np.array([1, 1, 1, 1, 1, 1, 1])
rdf = "td_fullrdf_mp"

if rdf == "td_fullrdf_1l" or "td_fullrdf_al" or "td_fullrdf_al_half":
    rdftitle = "Cluster RDF"
elif rdf == "td_fullrdf_raw":
    rdftitle = "Cluster RDF (No normalization)"
elif rdf == "td_simrdf":
    rdftitle = "PBC RDF"
elif rdf == "td_fullrdf_mp":
    rdftitle = "Cluster RDF (MP)"
name = str(ratio).replace(" ", "").replace("[", "").replace("]", "")
# samplename = f"packed_td/{rdf}_{name}_{binmax}_pbc.npy"
samplename = f"{rdf}/training_data_mp_1000_pbc_big.npy"

samples = np.load(samplename, allow_pickle=True)

X = np.array([i[0] for i in samples])
y = np.array([i[1] for i in samples])

# print(X)
# print(y)

def avg(bin = 1000):
    avg_peaknum = np.zeros(7, dtype=float)
    avg_peakdis = np.zeros((7, bin), dtype=float)
    avg_nonzerodis = np.zeros((7, bin), dtype=float)
    distribution = np.zeros(7, dtype=float)

    for a, b in zip(X, y):
        # print(np.count_nonzero(a), np.argmax(b))
        avg_peaknum[np.argmax(b)] += np.count_nonzero(a)
        avg_peakdis[np.argmax(b)] += a
        avg_nonzerodis[np.argmax(b)] += np.histogram((a != 0).nonzero()[0], bins=np.arange(bin+1))[0]
        distribution[np.argmax(b)] += 1

    avg_peaknum /= distribution
    for i in range(7):
        avg_peakdis[i] /= distribution[i]
        avg_nonzerodis[i] /= distribution[i]
        # print(avg_peakdis[i][999])
    # avg_zeros /= distribution
    print(avg_peaknum)
    # print(avg_peakdis)
    return avg_peaknum, avg_peakdis, avg_nonzerodis

def plotavg(avg_peakdis, avg_nonzerodis, movavg_peakdis, movavg_nonzerodis, bin = 1000):
    fig = plt.figure(figsize=(18,6))
    fig.suptitle("Averaged RDF (MP)")
    for i in range(2):
        for j in range(4):
            if i == 1 and j == 3:
                break
            plt.subplot2grid((2, 4),(i, j))
            # plt.gca().set_ylim([0, np.max(avg_peakdis) * 1.1])
            plt.plot(np.arange(bin)/bin, avg_peakdis[i*4+j])
            # plt.plot(np.arange(len(movavg_peakdis[0])) / bin, movavg_peakdis[i * 4 + j],
            #          c="yellow", ls="--", lw=2.5)
            if i == 1:
                plt.xlabel("% Radial distance from RMAX")
            plt.title(f"{types[i*4+j]}")
    plt.show()

    fig3 = plt.figure(figsize=(18,6))
    fig3.suptitle("Non-zero bins distribution (MP)")

    for i in range(2):
        for j in range(4):
            if i == 1 and j == 3:
                break
            plt.subplot2grid((2, 4),(i, j))
            plt.gca().set_ylim([0, 1])
            plt.plot(np.arange(bin)/bin, avg_nonzerodis[i*4+j])
            # plt.plot(np.arange(len(movavg_nonzerodis[0])) / bin, movavg_nonzerodis[i * 4 + j],
            #          c="orange", alpha=1, lw=2.5)
            if i == 1:
                plt.xlabel("% Radial distance from RMAX")
            if j == 0:
                plt.ylabel("% of datum")
            plt.title(f"{types[i*4+j]}")
    plt.show()

# n is better too be odd
def movavg(avg_peakdis, avg_nonzerodis, bin=1000, n=20):
    movavg_peakdis = np.zeros((7, bin-n+1), dtype=float)
    movavg_nonzerodis = np.zeros((7, bin-n+1), dtype=float)
    for i in range(len(avg_peakdis[0])):
        if i+n > len(avg_peakdis[0]):
            break
        movavg_peakdis[:, i] = np.mean(avg_peakdis[:, i:i+n], axis=1)
        movavg_nonzerodis[:, i] = np.mean(avg_nonzerodis[:, i:i+n], axis=1)
    return movavg_peakdis, movavg_nonzerodis

avg_peaknum, avg_peakdis, avg_nonzerodis = avg()
movavg_peakdis, movavg_nonzerodis = movavg(avg_peakdis, avg_nonzerodis)
plotavg(avg_peakdis, avg_nonzerodis, movavg_peakdis, movavg_nonzerodis)