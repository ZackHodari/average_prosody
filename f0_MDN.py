import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana.metrics import LF0Distortion
from morgana.viz.synthesis import MLPG
from morgana import data
from morgana import losses
from morgana import utils

from misc import batch_synth, GaussianMixtureModel

from tts_data_tools import data_sources


class F0_MDN(BaseSPSS):
    def __init__(self, input_dim=600+9, output_dim=1*3, dropout_prob=0., mixture_components=4, var_floor=1e-4):
        """Initialises acoustic model parameters and settings."""
        super(F0_MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.n_components = mixture_components
        self.var_floor = var_floor

        self.recurrent_layers = utils.SequentialWithRecurrent(
            nn.Linear(self.input_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(256, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, (self.output_dim * 2 + 1) * self.n_components),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=LF0Distortion())

    def normaliser_sources(self):
        return {
            'dur': data.MeanVarianceNormaliser('dur'),
            'lab': data.MinMaxNormaliser('lab'),
            'counters': data.MinMaxNormaliser('counters'),
            'lf0': data.MeanVarianceNormaliser('lf0', use_deltas=True),
        }

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames', sentence_level=True),
            'dur': data_sources.TextSource('dur'),
            'lab': data_sources.NumpyBinarySource('lab'),
            'counters': data_sources.NumpyBinarySource('counters'),
            'lf0': data_sources.NumpyBinarySource('lf0', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
        }

    def valid_data_sources(self):
        sources = self.train_data_sources()
        sources['mcep'] = data_sources.NumpyBinarySource('mcep')
        sources['bap'] = data_sources.NumpyBinarySource('bap')

        return sources

    def predict(self, features):
        # Prepare inputs.
        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(norm_lab, dur)

        norm_counters = features['normalised_counters']
        model_inputs = torch.cat((norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the encoder.
        n_frames = features['n_frames']
        predicted = self.recurrent_layers(model_inputs, seq_len=n_frames)

        # Extract the mixing components, means, and standard deviations from the prediction.
        i, j = self.n_components, self.n_components * self.output_dim
        pi = predicted[..., :i]
        means = torch.split(predicted[..., i:i+j], self.output_dim, dim=-1)
        log_variances = torch.split(predicted[..., i+j:], self.output_dim, dim=-1)

        # Set a variance floor.
        log_variances = [torch.clamp(log_variance, min=np.log(self.var_floor)) for log_variance in log_variances]

        # Reparameterisation: mixing coefficients should be a distribution, standard deviation should be non-negative.
        pi = F.softmax(pi, dim=-1)
        std_devs = [torch.exp(log_variance * 0.5) for log_variance in log_variances]

        # Prepare the outputs.
        pred_norm_lf0_deltas_GMM = GaussianMixtureModel(pi, means, std_devs)

        # Take the most likely components and find the most probable trajectory using MLPG.
        pred_norm_lf0_deltas_mean, pred_norm_lf0_deltas_std_dev = pred_norm_lf0_deltas_GMM.argmax_components()
        pred_norm_lf0_deltas_var = pred_norm_lf0_deltas_std_dev ** 2

        pred_norm_lf0 = MLPG(means=pred_norm_lf0_deltas_mean, variances=pred_norm_lf0_deltas_var)
        pred_lf0 = self.normalisers['lf0'].denormalise(pred_norm_lf0)

        outputs = {
            'normalised_lf0_deltas_GMM': pred_norm_lf0_deltas_GMM,
            'lf0': pred_lf0
        }

        return outputs

    def loss(self, features, output_features):

        @losses.sequence_loss
        def NLL(predictions, targets):
            return -predictions.log_prob(targets)

        seq_len = features['n_frames']
        loss = NLL(output_features['normalised_lf0_deltas_GMM'], features['normalised_lf0_deltas'], seq_len)

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], features['vuv'], seq_len))

        return loss

    def analysis_for_valid_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        kwargs['sample_rate'] = sample_rate
        super(F0_MDN, self).analysis_for_valid_batch(features, output_features, out_dir, **kwargs)
        batch_synth(features, output_features, out_dir, sample_rate)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0_MDN, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

