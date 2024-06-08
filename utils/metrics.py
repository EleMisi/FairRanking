from typing import Union

import numpy as np
import torch





def impact_function(scores: np.ndarray, thetas: np.ndarray):
    """
    Compute impact function given the resource and pool of queries
    :param scores:
        pool of  score vectors
    :param thetas:
        vector of modifier for each resource
    :return:
        impact function value
    """
    # Compute actual score
    scores = (scores * (1 - thetas)).mean(0)

    return scores


class Metric:

    def __init__(self, actions:str):
        """

        :param actions:
            type of mitigation actions, it can be 'group_weights' or 'polynomial_fn'
        """

        self.scaling = 'None'
        self.actions = actions

    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray):
        """
        Computes metric value given batch of queries and theta vector
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parameters
        """
        NotImplementedError()

    def compute_modified_scores(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray, ):
        """
        Computes modified score according to pre-defined actions
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parameters
        """
        if self.actions == 'group_weights':
            # Define vector of theta based on sensitive attribute
            thetas = np.zeros(len(batch_escs))
            for k in range(len(batch_escs)):
                # Find resource country
                country = np.argmax(batch_escs_discr[k])
                # Select country multiplier
                thetas[k] = parameters[country]

            return batch_scores.ravel() * (1 - thetas)

        elif self.actions == 'polynomial_fn':
            # Evaluate polynomial fn in given points
            g = np.stack([batch_escs ** d - np.mean(batch_escs ** d) for d in np.arange(parameters.shape[0]) + 1],
                         axis=1)  # We want g polynomial with zero mean
            w_b = g @ parameters

            # Normalize weights in the batch
            return batch_scores * w_b

    def scale(self, value):

        if self.scaling == 'normalization':
            value = (value - self.min) / (self.max - self.min)
        elif self.scaling == 'standardization':
            value = ((value - self.mean) / self.std)
        elif self.scaling == 'IQR_normalization':
            value = (value - self.q1) / (self.q3 - self.q1)
        elif self.scaling == 'min_std':
            value = (value - self.min) / self.std
        elif self.scaling == 'None':
            return value

        return value


class DIDI(Metric):
    """
    DIDI metric computation for students task
    """

    def __init__(self, actions, impact_function):
        """
        Build DIDI object
        :param actions:
            type of mitigation actions, it can be 'group_weights' or 'polynomial_fn'
        :param impact_function:
            function to compute impact of resource in the ranking
        """

        super().__init__(actions)
        self.impact_function = impact_function

        # TODO: DEBUG
        self.folder = None

    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray):
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parameters
        :return:
            The (absolute) value of the DIDI.
        """

        # Define indicator matrix for pool of queries
        indicator_matrix = self.get_indicator_matrix(batch_escs)

        # Compute impact function which is coincident to the modified scores
        self.modified_scores = self.compute_modified_scores(batch_scores,batch_escs,batch_escs_discr,parameters)

        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == self.modified_scores.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {self.modified_scores.shape[0]}"
        # Compute DIDI
        didi = self.compute_DIDI(output=self.modified_scores, indicator_matrix=indicator_matrix)

        didi = self.scale(didi)

        return didi

    def compute_DIDI(self, output: np.array, indicator_matrix: np.array) -> float:
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param output:
            array with impact function values (ordered by reources idx)
        :return:
            The (absolute) value of the DIDI.
        """
        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {output.shape[0]}"

        # Compute DIDI
        didi = 0.0
        total_average = np.mean(output)
        # Loop over protected groups
        for protected_group in indicator_matrix:
            # Select output of sample belonging to protected attribute
            protected_targets = output[protected_group]
            # Compute partial DIDI over the protected attribute
            if len(protected_targets) > 0:
                protected_average = np.mean(protected_targets)
                didi += abs(protected_average - total_average)
        return didi

    def get_indicator_matrix(self, batch_escs_attribute: np.ndarray) -> np.array:
        """Computes the indicator matrix given the input data and a protected feature.
        :param batch_escs_attribute:
            vector of queries sensitive attribute value
        :return:
            indicator matrix, i.e., a matrix in which the i-th row represents a boolean vector stating whether or
            not the j-th sample (represented by the j-th column) is part of the i-th protected group.
        """
        n_samples = batch_escs_attribute.shape[0]
        protected_labels = range(batch_escs_attribute.shape[1])
        n_groups = len(protected_labels)
        matrix = np.zeros((n_samples, n_groups)).astype(int)
        for i in range(n_groups):
            for j in range(n_samples):
                label = protected_labels[i]
                matrix[j, i] = 1 if batch_escs_attribute[j, label] == 1. else 0
        return matrix.transpose().astype(bool)



class GeDI(Metric):
    """
    GeDI metric computation for students task.
    Code adapted from the codebase of "Generalized Disparate Impact for Configurable Fairness Solutions in ML":
    https://github.com/giuluck/GeneralizedDisparateImpact
    """

    def __init__(self,
                 actions : str,
                 degree: int = 1,
                 relative: Union[bool, int] = 1,
                 ):
        """
        :param degree:
            The kernel degree for the excluded feature.

        :param relative:
            If a positive integer k is passed, it computes the relative value with respect to the indicator computed on
            the original targets with kernel k. If True is passed, it assumes k = 1. Otherwise, if False is passed, it
            simply computes the absolute value of the indicator.

        """
        super().__init__(actions)



        self.degree: int = degree
        """The kernel degree used for the features to be excluded."""

        self.relative: int = int(relative) if isinstance(relative, bool) else relative
        """The kernel degree to use to compute the metric in relative value, or 0 for absolute value."""







    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray):
        """Computes the GeDI given batch of scores and escs
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parametersve attribute
        :return:
            The (absolute) value of the GeDI.
        """

        # Modify the scores
        self.modified_scores = self.compute_modified_scores(batch_scores,batch_escs, batch_escs_discr, parameters)

        x = batch_escs

        # Compute GeDI
        gedi_y = self.compute_GeDI(x, y=self.modified_scores, degree=self.degree)
        gedi = gedi_y if gedi_y > 0 else 0.0

        # FIXME: scaling??
        gedi = self.scale(gedi)

        return gedi

    def compute_GeDI(self, x, y, degree: int, use_torch: bool = False):
        """
        Computes the generalized didi as the norm 1 of the vector <alpha_tilde> which is the solution of the following
        least-square problem:
            argmin || <phi> @ <alpha_tilde> - <psi> ||_2^2
        where <phi> is the zero-centered kernel matrix built from the excluded vector x, and <psi> is the zero-centered
        constant term vector built from the output targets y.

        :param x:
            The vector of features to be excluded.

        :param y:
            The vector of output targets.

        :param degree:
            The kernel degree for the features to be excluded.

        :param use_torch:
            Whether to compute the weights using torch.lstsq or numpy.lstsq

        :return:
            The value of the generalized didi.
        """
        alpha_tilde = GeDI.get_weights(x=x, y=y, degree=degree, use_torch=use_torch)
        didi = torch.abs(alpha_tilde).sum() if use_torch else np.abs(alpha_tilde).sum()

        assert didi >= 0

        return didi

    @staticmethod
    def get_weights(x, y, degree: int, use_torch: bool = False):
        """
        Computes the the vector <alpha> which is the solution of the following least-square problem:
            argmin || <phi> @ <alpha_tilde> - <psi> ||_2^2
        where <phi> is the zero-centered kernel matrix built from the excluded vector x, and <psi> is the zero-centered
        constant term vector built from the output targets y.

        :param x:
            The vector of features to be excluded.

        :param y:
            The vector of output targets.

        :param degree:
            The kernel degree for the features to be excluded.

        :param use_torch:
            Whether to compute the weights using torch.lstsq or numpy.lstsq

        :return:
            The value of the generalized didi. If return_weights is True, a tuple (<didi>, <alpha_tilde>) is returned.
        """
        phi = [x ** d - (x ** d).mean() for d in np.arange(degree) + 1]
        psi = y - y.mean()
        if use_torch:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            phi = torch.stack(phi, dim=1)
            alpha, _, _, _ = torch.linalg.lstsq(phi, psi, driver='gelsd')
        else:
            phi = np.stack(phi, axis=1)
            alpha, _, _, _ = np.linalg.lstsq(phi, psi, rcond=None)
        return alpha



class MeanSquaredError(Metric):




    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray):
        """
        Compute distance between true scores and modified scores
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parameters

        """

        self.modified_scores = self.compute_modified_scores(batch_scores=batch_scores, batch_escs =batch_escs, batch_escs_discr=batch_escs_discr, parameters=parameters)

        # Squared distance
        distance = np.power(batch_scores - self.modified_scores,2)

        # Mean
        distance =  distance.mean()

        # Re-scaling
        distance = self.scale(distance)

        return distance


class MeanAbsoluteError(Metric):




    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray, parameters: np.ndarray):
        """
        Compute distance between true scores and modified scores
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param parameters:
            vector of actions parameters

        """

        self.modified_scores = self.compute_modified_scores(batch_scores=batch_scores, batch_escs =batch_escs, batch_escs_discr=batch_escs_discr, parameters=parameters)

        # Distance
        distance = batch_scores - self.modified_scores

        # Mean
        distance =  distance.mean()

        # Re-scaling
        distance = self.scale(distance)



        return distance
