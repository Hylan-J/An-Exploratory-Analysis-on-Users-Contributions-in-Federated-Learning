import itertools

import numpy as np


def get_clients_subsets(n_client):
    client_list = list(np.arange(n_client))
    set_of_all_subsets = set([])
    for i in range(len(client_list), -1, -1):
        for element in itertools.combinations(client_list, i):
            set_of_all_subsets.add(frozenset(element))
    return sorted(set_of_all_subsets)
