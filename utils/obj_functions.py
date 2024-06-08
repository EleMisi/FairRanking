import numpy as np


class ObjFun():
    def __init__(self, metrics_dict: dict, metrics_weight: dict, n_params: int, scaling: str):
        """
        :param metrics:
            dict of Metric object
        :param args_metrics:
            additional arguments of the fairness metrics
        :param metrics_weight:
            dict of weights to aggregate the metrics
        """

        self.metrics_dict = metrics_dict
        self.metrics_weight = metrics_weight
        self.n_metrics = len(metrics_dict)
        self.n_params = n_params
        self.scaling = scaling

    def __call__(self, parameters: np.array):
        NotImplementedError()


class ObjFun_Distancex0():

    def __init__(self, x0: np.array):
        self.x0 = x0 # Point to be projected

    def __call__(self,x: np.array):
        dist = (x-self.x0) ** 2
        return dist.sum()

class ObjFun_DynStateVar(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray,  batch_escs_discr: np.ndarray, y: np.ndarray):
        """

        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param y:
            var to approximate

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.batch_escs_discr = batch_escs_discr
        self.y = y

    def __call__(self, params: np.array):
        """
        Compute the squared error between real synamic state and approximated one.
        :param params:
            [vector of modifiers for country, var_to_approximate]

        :return:
            approximation error
        """

        # Distinguish between theta variable and threshold variables
        thr_vars = params[self.n_params:]
        parameters = params[:self.n_params]
        y = np.zeros_like(self.y)

        j = 0
        # Assign variable value
        for k, value in enumerate(self.y):
            # if == -1, then the dynamic state was less than the threshold and we want to be free to change it to improve the metrics
            if value == -1:
                y[k] = thr_vars[j]
                j += 1
            # else, the dynamic state was greater than the threshold and we want to move towards it
            else:
                y[k] = value

        # Compute x approximate
        x_approx = np.zeros((self.n_metrics,))
        weights = np.zeros((self.n_metrics,))
        for i, name in enumerate(self.metrics_dict):
            x_approx[i] = self.metrics_dict[name](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                                  batch_escs_discr = self.batch_escs_discr,
                                                  parameters=parameters)
            # Populate weight vector
            weights[i] = self.metrics_weight[name]


        # Compute distance
        diff = x_approx - y
        # Square diff
        squared_diff = diff ** 2
        # Weighted sum
        cost = np.sum(squared_diff * weights) ** 0.5
        return cost


class ObjFun_DynState(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray, batch_escs_discr: np.ndarray,
                           x: np.array):
        """

        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param x:
            dynamical state to approximate

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.batch_escs_discr = batch_escs_discr
        self.y = x

    def __call__(self, parameters: np.array):
        """
        Compute the squared error between real synamic state and approximated one.
        :param parameters:
            vector of modifiers for country

        :return:
            approximation error
        """
        # Compute x approximate
        x_approx = np.zeros_like(self.x)
        weights = np.zeros((self.n_metrics,))
        for i, name in enumerate(self.metrics_dict):
            x_approx[i] = self.metrics_dict[name](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                                  parameters=parameters)
            # Populate weight vector
            weights[i] = self.metrics_weight[name]

        # Compute distance
        diff = x_approx - self.x
        # Square diff
        squared_diff = diff ** 2
        # Weighted sum
        cost = np.sum(squared_diff * weights) ** 0.5
        return cost


class ObjFun_BaselineSum(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray,  batch_escs_discr: np.ndarray, thresholds: dict):
        """

        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param thresholds:
            dict with metrics thresholds

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.batch_escs_discr = batch_escs_discr
        self.thresholds = thresholds

    def __call__(self, parameters: np.array):
        """
        Sum of metrics as cost function.
        :param parameters:
            vector of modifiers for country

        :return:
            sum of max(metric, threshold)
        """
        # Compute metrics
        metrics = np.zeros((self.n_metrics,))
        weights = np.zeros((self.n_metrics,))


        for i, name in enumerate(self.metrics_dict):
            # Max between metrics value and threshold
            metric = self.metrics_dict[name](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                             batch_escs_discr=self.batch_escs_discr,
                                             parameters=parameters)
            metrics[i] = max(metric, self.thresholds[i])
            # Populate weight vector
            weights[i] = self.metrics_weight[name]


        cost = (metrics * weights).sum()
        return cost


class ObjFun_Normalization(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray,  batch_escs_discr: np.ndarray, target_metric: str):
        """

        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param batch_escs_discr:
             discretized escs of current batch
        :param target_metric:
            name of the metric to minimize

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.batch_escs_discr = batch_escs_discr
        self.target_metric = target_metric

    def __call__(self, parameters: np.array):
        """
        Computes target metric value given theta vector
        :param parameters:
            vector of modifiers for country
        :return:
            sum of max(metric, threshold)
        """
        cost = self.metrics_dict[self.target_metric](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                                     batch_escs_discr=self.batch_escs_discr,
                                                     parameters=parameters)
        return cost
