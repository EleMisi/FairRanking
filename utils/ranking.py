from collections import OrderedDict

import numpy as np


class Ranking():

    def rank(self, queries: dict, theta_vector: np.array):
        """
        Rank based on query_scores
        :param queries:
            queries scores and sensitive attribute
        :param theta_vector:
            vector with modifiers for each
        :return:
            list of ascending ordered (resource_identifier, score) for each query

        """

        queries_scores = queries['score_MAT']

        # Define indicator matrix for pool of queries
        queries_sensitive = queries['ESCS']

        # Define vector of theta based on sensitive attribute
        thetas = np.zeros(len(queries_sensitive))
        for k in range(len(queries_sensitive)):
            # Find resource country
            country = np.argmax(queries_sensitive[k])
            # Select country multiplier
            thetas[k] = theta_vector[country]

        # Sort resource indexes based on the scores
        resource_idxs = np.argsort(-queries_scores)

        # List of rankings
        ordered_matching_list = []
        for q in range(len(queries_scores)):
            # Return list of (resource_identifier, matching_score)
            ordered_matching = [(idx, queries_scores[q][idx])
                                for idx in resource_idxs[q]]
            ordered_matching_list.append(OrderedDict(ordered_matching))

        return ordered_matching_list
