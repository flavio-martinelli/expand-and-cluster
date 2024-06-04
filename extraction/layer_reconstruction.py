"""
 # Created on 18.10.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Defines the main processes of the Expand-and-Cluster algorithm
 #
"""
import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from extraction.plotting import plot_all_features
from training.wandb_init import log_figure_wandb


def reconstruct_layer(W, N, gamma, beta, losses, A=None, dist="L2", final_layer=False, symmetry="none",
                      cluster_mask=None, verbose=False, plots_folder=None, exp_name=""):
    """
    Performs expand-and-cluster on a layer matrix
    :param W: Numpy tensor of layer l weight matrices of size [fan_in, fan_out, N] or [width*height, #channels,
    N] for convolutions.
    :param N: Number of students trained.
    :param gamma: Small cluster threshold is \gamma \cdot N.
    :param beta: Alignment threshold angle in radians.
    :param losses: Array of losses of each student used to choose the element to collapse the cluster into.
    :param A: Numpy output weights tensor [fan_out, next layer fan_in, N]. Used only to merge output weights with the
    clusters, if set to None then no output weights are returned.
    :param dist: if "cosine" uses cosine similarity, otherwise L2 pairwise distance.
    :param final_layer: if True, returns the output weights corresponding to the selected student neurons (not all
    of them)
    :param symmetry: symmetries can be 'none', 'odd', 'even_linear', 'even_linear_positive_scaling', 'odd_constant'
    :param cluster_mask: mask to apply to the vectors before clustering (e.g. Boruta mask)
    :param verbose: print some infos along the way.
    :param plots_folder: if not None, saves plots (tree, L2 matrix, ...) in that folder.
    :param exp_name: name used for the figure files. e.g. layer number or anything.
    :return: (\theta_l, \theta_{l+1}) tuple of reconstructed weight matrix of layer l and output weights for next layer.
    The latter is None if A=None.
    """
    assert N == W.shape[2], f"W tensor last dimension {W.shape[2]} and number of students N={N} do not match!"
    assert gamma * N >= 2, f"The small cluster threshold (gamma * N = {gamma * N}) is lower than 2!"
    if plots_folder is not None:
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

    fan_in = W.shape[0]
    fan_out = W.shape[1]
    features = copy.deepcopy(np.transpose(W, (2, 1, 0)).reshape(-1, fan_in))  # concatenate all neuron features from
    # different nets

    canonical_first_id = 0
    if cluster_mask is not None:  # Ensures the pixel is chosen for flipping is meaningful
        canonical_first_id = np.where(cluster_mask == 1)[0].min()

    original_signs = np.ones_like(features[:, canonical_first_id])
    if symmetry != 'none':  # Canonical transformation of the features.
        original_signs = np.sign(features[:, canonical_first_id])  # Keep which feature is flipped to recover later
        features = np.einsum('ij,i->ij', features, np.sign(features[:, canonical_first_id]))

    # Clustering step
    clusters, cluster_indices, H, simmat, _, h_thr = \
        find_clusters(features, int(gamma * N), linkage="average", max_clusters=True, dist=dist,
                      forced_cut_height= 1 - np.cos(beta) if dist=="cosine" else None, cluster_mask=cluster_mask,
                      verbose=True)

    big_cluster_no = len(clusters)
    if verbose:
        print(f"Found {big_cluster_no} of size bigger than {gamma*N}.")

    # Remove unaligned clusters
    clusters, cluster_indices, median_cos_distances = remove_unaligned_clusters(clusters, cluster_indices, beta,
                                                                                cluster_mask=cluster_mask,
                                                                                verbose=verbose)
    print(f"Collapsing {len(clusters)} clusters.")

    # Plot clustering results before collapsing
    if plots_folder is not None:
        if features.shape[0] < 15000:  # Plotting for bigger matrices crashes
            if verbose:
                print("producing dendrogram...")
            fig, ax = plot_dendrogram(H, color_threshold=h_thr)
            ax.axhline(h_thr, color="orange", linestyle="--")
            fig.savefig(os.path.join(plots_folder, f"{exp_name}_dendrogram.pdf"))
            plt.close(fig)

            if verbose:
                print("producing simmat...")
            fig, ax = plot_simmat(H, simmat)
            ax.set_title("")
            fig.savefig(os.path.join(plots_folder, f"{exp_name}_simmat.pdf"))
            plt.close(fig)

        fig, ax = plot_alignments(median_cos_distances, beta)
        ax.set_title(f"Cluster alignments: {len(clusters)} clusters found")
        # log_figure_wandb(fig, f"clustering/{exp_name}_alignments")
        fig.savefig(os.path.join(plots_folder, f"{exp_name}_alignments.pdf"))
        plt.close(fig)

    # Collapse clusters: theta = (list of w_reconstructed, list of a_reconstructed)
    theta = collapse(clusters, cluster_indices, W, A, N, losses, fan_out, final_layer, original_signs, symmetry,
                     verbose)
    w_reconstructed = np.array(theta[0]).T
    a_reconstructed = None
    # plot_all_features(w_reconstructed[:-1, :, np.newaxis])
    if A is not None:
        if final_layer:
            a_reconstructed = np.array(theta[1])
        else:
            a_reconstructed = np.transpose(np.array(theta[1]), [0, 2, 1])
    # plot_all_features(a_reconstructed[:, :, np.newaxis], reshape=False, vlims=[-1, 1])[1].show()
    return w_reconstructed, a_reconstructed


def collapse(clusters, cluster_indices, W, A, N, losses, fan_out, final_layer, original_signs, symmetry, verbose):
    """

    :param clusters:
    :param cluster_indices:
    :param W:
    :param A:
    :param N:
    :param losses:
    :param fan_out:
    :param verbose:
    :return:
    """

    best_losses = np.argsort(losses)[:5]  # best 5 students and respective indices
    w_centers = []
    cluster_outputs = []
    final_A = []

    if verbose:
        print(f"Collapsing based on best student neuron 'cluster number -> student rank':")

    # i_clus: cluster number
    # cluster: weight vectors in the cluster
    # indices: index of the weight vector in the flattened W matrix
    for i_clus, (cluster, indices) in enumerate(zip(clusters, cluster_indices)):
        w_center = 0
        w_normalizer = 0

        # keeps track of the output weights for different networks (accounting for permutations of the next hidden
        # layer). It is called a_sums cause duplicates will be merged into one summed contribution.
        if A is not None:
            a_sums = np.zeros([N, A.shape[1]])  # sum of output weights divided per network

        # find neuron indices of best student
        neurons_idx_best_student = []
        for best_no, nn in enumerate(best_losses):
            neurons_idx_best_student = np.arange(nn * fan_out, (nn + 1) * fan_out)
            if np.sum([i in neurons_idx_best_student for i in indices]) > 0:
                # the best_no^th best student contains neurons in this cluster
                if verbose:
                    print(f"#{i_clus}->{best_no+1}, ", end="")
                break

        # Loop through cluster elements and:
        # (i) find the ones belonging to neurons_idx_best_student and use them to construct a w_center (winner take
        # all policy)
        # (ii) keep track of output weights of different networks and sum them together if duplicates within same
        # networks are found

        for i in indices:
            net_idx = i // fan_out  # Neuron i belong to student number net_idx
            sign = original_signs[i]
            if A is not None:  # compute the output weights and their sums to account for duplicate neurons
                a = A[i - fan_out * net_idx, :, net_idx] # Output vector of neuron i
                if symmetry == 'odd' or symmetry == 'odd_constant':  # for even symmetries sign flip is not required
                    a = a * sign
                a_sums[net_idx] += a  # Add output vector to corresponding student index

            if i not in neurons_idx_best_student:  # if current student is not best: continue
                continue
            w = W[:, i - fan_out * net_idx, net_idx] * sign  # Feature vector of neuron i
            w_center += w  # In case of duplicate, compute an avg of features
            w_normalizer += 1  # (could also include magnitude of a to make a weighted avg)

        # save input vector of best student (averaged if duplicated)
        if np.isscalar(w_center):
            continue
        if w_normalizer != 0:
            w_centers.append(w_center / w_normalizer)
        else:
            w_centers.append(w_center)

        # save output vectors of all elements if not final layer, otherwise only of best student
        if A is not None and not final_layer:
            cluster_outputs.append(a_sums)
        elif final_layer:
            cluster_outputs.append(a_sums[nn, :])

    if verbose:
        print(f"\nFinal layer size: {len(w_centers)}\n")
    return w_centers, cluster_outputs


def remove_unaligned_clusters(C, C_idx, beta, cluster_mask=None, verbose=False):
    """

    :param C: list of clusters each containing >1 element
    :param C_idx: indices of these clusters from the find_cluster method
    :param beta: alignment threshold angle in radians.
    :param cluster_mask: mask to apply to the vectors before clustering (e.g. Boruta mask)
    :param verbose: verbosity.
    :return: clusters, cluster_indices, median_cos_dist
    """

    median_cos_distances = []
    for i, clus in enumerate(C):
        clus_masked = np.einsum('ij,j->ij', clus, cluster_mask) if cluster_mask is not None else clus
        cosmat = pdist(clus_masked, "cosine")
        if len(cosmat) == 0:
            continue
        median_cos_distances.append(np.median(cosmat))
    cos_dist_tol = 1 - np.cos(beta)
    unaligned_idxs = np.where(np.array(median_cos_distances) > cos_dist_tol)[0]
    if verbose:
        print(f"removing {len(unaligned_idxs)}/{len(C)} unaligned clusters ")
    clusters = [c for i, c in enumerate(C) if i not in unaligned_idxs]
    cluster_indices = [ci for i, ci in enumerate(C_idx) if i not in unaligned_idxs]
    return clusters, cluster_indices, median_cos_distances


def find_clusters(w, min_size, plateau_threshold=0.1, fix_clusters=None, linkage="average", max_clusters=False,
                  dist="L2", forced_cut_height=None, cluster_mask=None, verbose=False):
    """
    Extracts clusters given set of vectors with average linkage hierarchical clustering. The clusters are found by
    cutting the tree at the height corresponding to the first pleteau in the curve: #C > no_nets vs. height.
    :param w: matrix of size (m*no_nets, d_in)
    :param min_size: minimum size to be considered a big cluster
    :param plateau_threshold: the first plateau longer than plateau_threshold determines the height at which to cut the
    tree
    :param fix_clusters: if set, the tree is cut when #fix_clusters clusters are above min_size (so to select cluster
     size)
    :param linkage: choose between "average" or "complete" linkage (does average by default)
    :param max_clusters: if True, cut the tree where there are most clusters of size above min_size
    :param dist: if "cosine" uses cosine similarity, otherwise L2 pairwise distance.
    :param forced_cut_height: if not None, cuts the tree at the given height
    :param cluster_mask: mask to apply to the vectors before clustering (e.g. Boruta mask)
    :param verbose: verbosity True or False
    :return: List of weights belonging to each cluster, cluster_indices, linkage result H, dissimilarity matrix
    """

    if verbose: print(f"computing pairwise {dist} distances...", end="")
    w_masked = np.einsum('ij,j->ij', w, cluster_mask) if cluster_mask is not None else w
    simmat = pdist(w_masked, "cosine") if dist == "cosine" else pdist(w_masked)
    if verbose: print(" Done!")
    if linkage == "complete":
        H = hierarchy.complete(simmat)
    else:
        H = hierarchy.average(simmat)
    if verbose: print("Finding tree cut threshold")
    big_clusters = []
    for i, thr in enumerate(H[:, 2]):  # loop through all merging heights
        c = hierarchy.fcluster(H, thr, criterion='distance')
        _, counts = np.unique(c, return_counts=True)
        big_clusters.append(len(counts[counts >= min_size]))

    big_clusters = np.array(big_clusters)

    argmax_b = np.argmax(big_clusters)
    if min_size > 1:
        # limit analysis up to maximum clusters found
        # (not if the number of student used is 1 otherwise the maximum is at 0)
        max_b = np.max(big_clusters)
        max_points = np.where(big_clusters == max_b)[0]
        argmax_b = max_points[0]
        for i in range(1, len(max_points)):
            if max_points[i - 1] + 1 == max_points[i]:
                argmax_b = max_points[i]
            else:
                break
        big_clusters = big_clusters[:argmax_b + 2]

    # computing the idx at which the number of clusters change
    change_idx = []
    for i in range(big_clusters.size):
        if big_clusters[i] != big_clusters[i - 1] and i > 0:
            change_idx.append(i)

    change_idx = np.array(change_idx)
    plateau_lengths = H[change_idx[1:], 2] - H[change_idx[:-1], 2]

    if max_clusters is True:  # select maximum number of found clusters (cuts at first occurrence, the lowest height)
        h_thr = H[argmax_b, 2]
    else:
        # threshold selected as almost end of plateau (at 90%)
        if fix_clusters is None:
            try:
                h_bar_start_idx = np.argwhere(plateau_lengths > plateau_threshold)[0, 0]
                h_thr = .05 * H[change_idx[h_bar_start_idx], 2] + .95 * H[change_idx[h_bar_start_idx + 1], 2]
            except Exception:  # if no plateau found, take h where the clusters are maximum
                h_thr = H[np.argmax(big_clusters), 2]
        else:
            try:
                h_thr = 0.05 * H[np.where(big_clusters == fix_clusters)[0][0], 2] + \
                        0.95 * H[np.where(big_clusters == fix_clusters + 1)[0][0], 2]
            except Exception as e:
                print("The selected number of clusters was never reached in the tree")

    if verbose: print("Extracting clusters")
    if forced_cut_height is not None:
        h_thr = forced_cut_height
    c = hierarchy.fcluster(H, h_thr, criterion='distance')
    cluster_idx, sizes = np.unique(c, return_counts=True)
    bigcluster_idx = cluster_idx[sizes >= min_size]
    clusters = []
    cluster_indices = []
    for b in bigcluster_idx:
        cluster_w_idx = np.argwhere(c == b)[:, 0]
        cluster_indices.append(cluster_w_idx)
        clusters.append(w[cluster_w_idx])

    return clusters, cluster_indices, H, simmat, plateau_lengths, h_thr


def plot_dendrogram(H, color_threshold=None):
    sys.setrecursionlimit(100000)  # Avoids stackoverflow for big dendrograms.
    fig, ax = plt.subplots(1, 1, figsize=(20, 18), dpi=750)
    dd2 = hierarchy.dendrogram(H, ax=ax, count_sort="descending", distance_sort=False, color_threshold=color_threshold,
                               get_leaves=True)
    ax.set_title("average linkage")
    return fig, ax


def plot_simmat(H, simmat):
    sys.setrecursionlimit(100000)  # Avoids stackoverflow for big dendrograms.
    fig, ax = plt.subplots(1, 1, figsize=(80, 60), dpi=750)
    dd2 = hierarchy.dendrogram(H, ax=ax, count_sort="descending", distance_sort=False, color_threshold=0,
                               get_leaves=True)

    leaf_orders = np.array(dd2['leaves'])
    expanded_simmat = squareform(simmat)
    cmap = 'OrRd_r'
    color_norm = colors.LogNorm()
    leaf_mat = expanded_simmat[np.ix_(leaf_orders, leaf_orders)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 9), dpi=800)
    im = ax.imshow(leaf_mat, cmap=cmap, norm=color_norm)
    ax.set_title(f"L2 dissimilarity matrix - average linkage")
    ax.set_xlabel("$w_j$")
    ax.set_ylabel("$w_i$")
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_alignments(cos_distances, beta):
    clusters_id = np.argsort(cos_distances)
    cos_distances = np.sort(cos_distances)
    cos_dist_beta = 1 - np.cos(beta)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=500)
    ax.bar(range(len(cos_distances)), cos_distances)
    if np.min(cos_distances) < 0.01:
        ax.set_yscale("log")
    ax.axhline(1, linestyle="--", color="grey", alpha=0.5, label="90°")
    ax.axhline(cos_dist_beta, linestyle="--", color="red", alpha=0.5, label=f"beta={beta * 180 / np.pi:.3f}°")
    ax.set_xticks(range(len(cos_distances)))
    ax.set_xticklabels(clusters_id)
    # smaller font size of xticks
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(2)
    # rotate xticklabels
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    ax.set_yticks(np.concatenate((ax.get_yticks(), [1, cos_dist_beta])))
    ax.set_xlabel("Cluster id")
    ax.set_ylabel("Cosine distance")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=None, log=True, verbose=False):
    if cluster_mask is None:
        cluster_mask = 1
    if symmetry == 'odd' or symmetry == 'odd_constant':
        dist_fun_w = abs_cosine_dissimilarity
        dist_fun_a = abs_cosine_dissimilarity
    elif symmetry == 'even_linear' or symmetry == 'even_linear_positive_scaling':
        dist_fun_w = abs_cosine_dissimilarity
        dist_fun_a = cosine_dissimilarity
    else:
        dist_fun_w = cosine_dissimilarity
        dist_fun_a = cosine_dissimilarity

    wt = np.concatenate([wt, bt[np.newaxis, :]], axis=0)
    ws = np.concatenate([ws, bs[np.newaxis, :]], axis=0)

    # sort teacher weights from highest to lowest output neuron norm
    idx_sorted = np.argsort(np.linalg.norm(at, ord=2, axis=1))[::-1]
    wt = wt[:, idx_sorted].astype(np.double)  # We need higher precision for cosine similarity
    at = at[idx_sorted, :].astype(np.double)

    best_sims_w = []
    signs_w = []
    signs_a = []
    student_idx_matched = []
    best_sims_a = []
    for wtt, att in zip(wt.T, at):
        sim_w = np.array([dist_fun_w(cluster_mask * wtt, cluster_mask * wss) for wss in ws.T])
        best_sims_w.append(sim_w.min())
        student_idx_matched.append(sim_w.argmin())
        sim_a = dist_fun_a(att, as_[sim_w.argmin()])
        signs_w.append(np.sign(1 - cosine_dissimilarity(cluster_mask * wtt, cluster_mask * ws.T[sim_w.argmin()])))
        signs_a.append(np.sign(1 - cosine_dissimilarity(att, as_[sim_w.argmin()])))
        best_sims_a.append(sim_a)

    teacher_best_sims_w = best_sims_w
    teacher_best_sims_a = best_sims_a

    if verbose:
        print(f"Best average sim w: {np.mean(best_sims_w)}")
        print(f"Best max sim w: {np.max(best_sims_w)}")
        print(f"Best average sim a: {np.mean(best_sims_a)}")
        print(f"Best max sim a: {np.max(best_sims_a)}")

    fig = plt.figure(figsize=(7, 5), dpi=200)
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[:, 3])

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=200, sharex=True)
    ax1.bar(range(len(best_sims_w)), best_sims_w)
    for i, patch in enumerate(ax1.patches):
        patch.set_facecolor("tab:blue" if signs_w[i] > 0 else "red")
    ax2.bar(range(len(best_sims_a)), best_sims_a)
    for i, patch in enumerate(ax2.patches):
        patch.set_facecolor("tab:blue" if signs_a[i] > 0 else "red")
    ax1.set_title("Distance from teacher neurons")
    ax1.set_ylabel("dist( w_t , w_s )")
    ax2.set_xlabel("Teacher neuron index")
    ax2.set_ylabel("dist( a_t , a_s )")
    if log:
        ax1.set_ylim([1e-6, 1])
        ax2.set_ylim([1e-6, 1])
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    else:
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

    # disable top and right spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.invert_yaxis()
    ax2.xaxis.tick_top()
    ax2.set_xticklabels([])

    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # Looking at the missing student neurons
    missing_students_idx = set(range(as_.shape[0])) - set(student_idx_matched)
    best_sims_w = []
    teacher_idx_matched = []
    student_out_norms = []
    for s_id in missing_students_idx:
        wss = ws.T[s_id]
        sim_w = np.array([dist_fun_w(cluster_mask * wtt, cluster_mask * wss) for wtt in wt.T])
        best_sims_w.append(sim_w.min())
        teacher_idx_matched.append(sim_w.argmin())
        student_out_norms.append(np.linalg.norm(as_[s_id], ord=2))

    # sort teacher weights from highest to lowest output neuron norm
    idx_sorted = np.argsort(teacher_idx_matched)
    teacher_idx_matched = np.array(teacher_idx_matched)[idx_sorted]
    student_out_norms = np.array(student_out_norms)[idx_sorted]
    excess_neurons = len(teacher_idx_matched)
    best_sims_w = np.array(best_sims_w)[idx_sorted]
    facecolors = ["C0" if n > 0.1 else "white" for n in student_out_norms]
    linestyles = ["-" if n > 0.1 else "--" for n in student_out_norms]

    if len(best_sims_w) < 10:
        best_sims_w = np.concatenate([best_sims_w, np.zeros(10 - len(best_sims_w))])
        teacher_idx_matched = np.concatenate([teacher_idx_matched, np.zeros(10 - len(teacher_idx_matched))-1]).astype(
            int)
        facecolors = np.concatenate([facecolors, ["white"] * (10 - len(facecolors))])
        linestyles = np.concatenate([linestyles, ["--"] * (10 - len(linestyles))])

    ax3.barh(range(len(teacher_idx_matched)), best_sims_w, edgecolor="C0")
    # change each bar's facecolor based on facecolors
    for i, patch in enumerate(ax3.patches):
        patch.set_facecolor(facecolors[i])
        patch.set_linestyle(linestyles[i])
    ax3.set_title(f"{excess_neurons} excess neurons")
    ax3.yaxis.tick_right()
    ax3.set_yticks(range(len(teacher_idx_matched)))
    ax3.set_yticklabels([l if l != -1 else '' for l in teacher_idx_matched])
    ax3.set_xlabel("dist( w_t , w_s )")
    ax3.set_ylabel("Teacher neuron index")
    ax3.yaxis.set_label_position("right")

    if log:
        ax3.set_xlim([1e-6, 1])
        ax3.set_xscale("log")
    else:
        ax3.set_xlim([0, 1])
    ax3.invert_xaxis()

    fig.tight_layout()
    return fig, teacher_best_sims_w, teacher_best_sims_a, (ws.shape[1], len(best_sims_w))


def compare_with_teacher_conv(wt, bt, ws, bs, symmetry, verbose=True, log=False):
    if symmetry == 'odd' or symmetry == 'odd_constant':
        dist_fun_w = abs_cosine_dissimilarity
    elif symmetry == 'even_linear' or symmetry == 'even_linear_positive_scaling':
        dist_fun_w = abs_cosine_dissimilarity
    else:
        dist_fun_w = cosine_dissimilarity

    wt = np.concatenate([wt, bt[np.newaxis, :]], axis=0)
    ws = np.concatenate([ws, bs[np.newaxis, :]], axis=0)
    wt = wt.astype(np.double)  # We need higher precision for cosine similarity

    best_sims_w = []
    signs_w = []
    student_idx_matched = []
    for wtt in wt.T:
        sim_w = np.array([dist_fun_w(wtt, wss) for wss in ws.T])
        best_sims_w.append(sim_w.min())
        student_idx_matched.append(sim_w.argmin())
        signs_w.append(np.sign(1 - cosine_dissimilarity(wtt, ws.T[sim_w.argmin()])))

    teacher_best_sims_w = best_sims_w

    if verbose:
        print(f"Best average sim w: {np.mean(best_sims_w)}")
        print(f"Best max sim w: {np.max(best_sims_w)}")

    fig = plt.figure(figsize=(7, 5), dpi=200)
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[:, 3])

    ax1.bar(range(len(best_sims_w)), best_sims_w)
    for i, patch in enumerate(ax1.patches):
        patch.set_facecolor("tab:blue" if signs_w[i] > 0 else "red")
    ax1.set_title("Distance from teacher neurons")
    ax1.set_ylabel("dist( w_t , w_s )")
    if log:
        if np.mean(best_sims_w) < 1e-6:
            ax1.set_ylim([1e-10, 1])
        else:
            ax1.set_ylim([1e-6, 1])
        ax1.set_yscale("log")
    else:
        ax1.set_ylim([0, 1])

    # disable top and right spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.invert_yaxis()
    ax2.xaxis.tick_top()
    ax2.set_xticklabels([])

    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # Looking at the missing student neurons
    missing_students_idx = set(range(ws.shape[1])) - set(student_idx_matched)
    best_sims_w = []
    teacher_idx_matched = []
    for s_id in missing_students_idx:
        wss = ws.T[s_id]
        sim_w = np.array([dist_fun_w(wtt, wss) for wtt in wt.T])
        best_sims_w.append(sim_w.min())
        teacher_idx_matched.append(sim_w.argmin())

    # sort teacher weights from highest to lowest output neuron norm
    idx_sorted = np.argsort(teacher_idx_matched)
    teacher_idx_matched = np.array(teacher_idx_matched)[idx_sorted]
    excess_neurons = len(teacher_idx_matched)
    best_sims_w = np.array(best_sims_w)[idx_sorted]

    if len(best_sims_w) < 10:
        best_sims_w = np.concatenate([best_sims_w, np.zeros(10 - len(best_sims_w))])
        teacher_idx_matched = np.concatenate([teacher_idx_matched, np.zeros(10 - len(teacher_idx_matched))-1]).astype(
            int)

    ax3.barh(range(len(teacher_idx_matched)), best_sims_w, edgecolor="C0")
    ax3.set_title(f"{excess_neurons} excess neurons")
    ax3.yaxis.tick_right()
    ax3.set_yticks(range(len(teacher_idx_matched)))
    ax3.set_yticklabels([l if l != -1 else '' for l in teacher_idx_matched])
    ax3.set_xlabel("dist( w_t , w_s )")
    ax3.set_ylabel("Teacher neuron index")
    ax3.yaxis.set_label_position("right")

    if log:
        ax3.set_xlim([1e-6, 1])
        ax3.set_xscale("log")
    else:
        ax3.set_xlim([0, 1])
    ax3.invert_xaxis()

    fig.tight_layout()
    return fig, teacher_best_sims_w, student_idx_matched, (ws.shape[1], len(best_sims_w))



def cosine_dissimilarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:  # if one of the vectors is 0, return orthogonal
        return 1
    return np.clip(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 1e-16, 2)


def abs_cosine_dissimilarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:  # if one of the vectors is 0, return orthogonal
        return 1
    return np.clip(1 - np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 1e-16, 1)
