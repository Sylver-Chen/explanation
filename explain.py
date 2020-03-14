

class Explanation(object):
    """Object returned by explainers."""

    def __init__(self,
                 # domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):
        """

        Initializer.

        Args:
            domain_mapper: must inherit from DomainMapper class
            type: "classification" or "regression"
            class_names: list of class names (only used for classification)
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.random_state = random_state
        self.mode = mode
        self.local_exp = {}
        self.intercept = {}
        self.score = None
        self.local_pred = None
        self.right = None
        self.scaled_data = None
        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None

