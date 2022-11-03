import torch
import torch.nn.functional as F

def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def flatten(lists):
    return [item for l in lists for item in l]

def fill_expand(labels):
    event_num = max(flatten(labels)) + 1
    filled_labels = torch.zeros((event_num, event_num)) # account for dummy
    for gr in labels:
        if len(gr) > 1:
            sorted_gr = sorted(gr)
            for i in range(len(sorted_gr)):
                for j in range(i+1, len(sorted_gr)):
                    filled_labels[sorted_gr[j]][sorted_gr[i]] = 1
        else:
            try:
                filled_labels[gr[0]][gr[0]] = 1 # dummy default to same index as itself
            except:
                print(gr)
                raise ValueError
    return filled_labels

def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    sizes = [s.shape[0] for s in tensors]
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes

def get_event2cluster(clusters):
    event2cluster = {}
    for cluster in clusters:
        for eid in cluster:
            event2cluster[eid] = tuple(cluster)
    return event2cluster

def get_clusters(event2cluster):
    clusters = list(set(event2cluster.values()))
    return clusters

def get_predicted_clusters(prob):
    predicted_antecedents = torch.argmax(prob, dim=-1).cpu().numpy().tolist() # n_event x n_event -> n_event
    idx_to_clusters = {}
    predicted_clusters = []
    for i in range(len(predicted_antecedents)):
        idx_to_clusters[i] = set([i])

    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index >= i:
            assert predicted_index == i
            continue
        else:
            union_cluster = idx_to_clusters[predicted_index] | idx_to_clusters[i]
            for j in union_cluster:
                idx_to_clusters[j] = union_cluster
    idx_to_clusters = {i: tuple(sorted(idx_to_clusters[i])) for i in idx_to_clusters}
    predicted_clusters = get_clusters(idx_to_clusters)
    return predicted_clusters, idx_to_clusters


if __name__ == "__main__":
    clus = [[1,2,3], [4,5,6], [7], [8, 10], [9]]
    print(fill_expand(clus))