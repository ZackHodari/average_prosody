import torch

from morgana.experiment_builder import ExperimentBuilder

from f0_RNN import F0_RNN


class F0_RNN_Scaled(F0_RNN):
    def __init__(self, normalisers=None, dropout_prob=0., scale=3):
        """Initialises acoustic model parameters and settings."""
        super(F0_RNN_Scaled, self).__init__(normalisers=normalisers, dropout_prob=dropout_prob)

        self.scale = scale

    def predict(self, features):
        outputs = super(F0_RNN_Scaled, self).predict(features)
        pred_lf0 = outputs['lf0']

        # Modify the predicted LF0 contour.
        pred_lf0_mean = torch.mean(pred_lf0, dim=0)
        lf0_residual = pred_lf0 - pred_lf0_mean
        pred_lf0_scaled = pred_lf0_mean + self.scale * lf0_residual

        outputs['lf0'] = pred_lf0_scaled

        return outputs


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0_RNN_Scaled, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

