class ComposePerturbator:
    def __init__(self, perturbator_list):
        self.perturbator_list = perturbator_list

    def __call__(self, x):
        y = x
        self.used_perturbation_list = []
        for perturbator_instance in self.perturbator_list:
            y = perturbator_instance(y)
        return y
