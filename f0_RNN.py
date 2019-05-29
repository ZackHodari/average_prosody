import os

import numpy as np
import torch
import torch.nn as nn

from morgana.base_models import BaseAcousticModel
from morgana.experiment_builder import ExperimentBuilder
from morgana.metrics import LF0Distortion
from morgana.synthesis import MLPG
from morgana import data
from morgana import utils

import tts_data_tools as tdt


class F0_RNN(BaseAcousticModel):
    def __init__(self, normalisers=None, dropout_prob=0.):
        """Initialises acoustic model parameters and settings."""
        super(F0_RNN, self).__init__(normalisers=normalisers)

        self.recurrent_layers = utils.SequentialWithRecurrent(
            nn.Linear(self.conditioning_dim, 256),
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
            nn.Linear(64, self.output_dim),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=LF0Distortion())

    @classmethod
    def train_data_sources(cls):
        return {
            'n_frames': data.TextSource('n_frames'),
            'n_phones': data.TextSource('n_phones'),
            'dur': data.TextSource('dur', normalisation='mvn'),
            'lab': data.NumpyBinarySource('lab', normalisation='minmax'),
            'counters': data.NumpyBinarySource('counters', normalisation='minmax'),
            'lf0': data.NumpyBinarySource('lf0', normalisation='mvn', use_deltas=True),
            'vuv': data.NumpyBinarySource('vuv', dtype=np.bool),
        }

    @classmethod
    def valid_data_sources(cls):
        data_sources = cls.train_data_sources()
        data_sources['sp'] = data.NumpyBinarySource('sp')
        data_sources['ap'] = data.NumpyBinarySource('ap')

        return data_sources

    @property
    def output_dim(self):
        return 1 * 3

    @property
    def conditioning_dim(self):
        return 600 + 9

    def predict(self, features):
        # Prepare inputs.
        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_durations(norm_lab, dur)

        norm_counters = features['normalised_counters']
        model_inputs = torch.cat((norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the encoder.
        n_frames = features['n_frames']
        pred_norm_lf0_deltas, _ = self.recurrent_layers(model_inputs, seq_len=n_frames)

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        device = pred_lf0_deltas.device
        pred_lf0 = MLPG.generate(pred_lf0_deltas.detach().cpu().numpy(),
                                 self.normalisers['lf0'].delta_params['std_dev'] ** 2)

        pred_lf0 = torch.tensor(pred_lf0).type(torch.float).to(device)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
            'lf0_deltas': pred_lf0_deltas,
            'lf0': pred_lf0
        }

        return outputs

    def loss(self, features, output_features):
        inputs = features['normalised_lf0_deltas']
        outputs = output_features['normalised_lf0_deltas']
        seq_len = features['n_frames']

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], seq_len, features['vuv']))

        return self._loss(inputs, outputs, seq_len)

    def analysis_for_eval_batch(self, output_features, features, names, out_dir, sample_rate=16000):
        super(F0_RNN, self).analysis_for_eval_batch(output_features, features, names, out_dir, sample_rate)

        lf0 = output_features['lf0'].cpu().detach().numpy()
        n_frames = features['n_frames'].cpu().detach().numpy()

        raw_dir = os.path.join(out_dir, 'raw', 'lf0')
        os.makedirs(raw_dir, exist_ok=True)
        for i, name in enumerate(names):
            feat_path = os.path.join(raw_dir, '{}.lf0'.format(name))
            tdt.file_io.save_bin(lf0[i, :n_frames[i]], feat_path)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0_RNN, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

