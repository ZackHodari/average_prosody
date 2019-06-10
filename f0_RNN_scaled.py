import torch

from morgana.experiment_builder import ExperimentBuilder

from f0_RNN import F0_RNN


class F0_RNN_Scaled(F0_RNN):
    def __init__(self, input_dim=600+9, output_dim=1*3, dropout_prob=0., scale=3):
        """Initialises acoustic model parameters and settings."""
        super(F0_RNN_Scaled, self).__init__(input_dim=input_dim, output_dim=output_dim, dropout_prob=dropout_prob)
        self.scale = scale

    def predict(self, features):
        outputs = super(F0_RNN_Scaled, self).predict(features)
        pred_lf0 = outputs['lf0']

        # Modify the predicted LF0 contour.
        pred_lf0_mean = torch.mean(pred_lf0, dim=0)
        lf0_residual = pred_lf0 - pred_lf0_mean
        pred_lf0_scaled = pred_lf0_mean + self.scale * lf0_residual

        # The loss is defined based on 'normalised_lf0_deltas' so this scaling will not interact with model training.
        outputs['lf0'] = pred_lf0_scaled

        return outputs


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0_RNN_Scaled, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

