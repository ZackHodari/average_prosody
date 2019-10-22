import os

import torch
import torch.nn as nn

from morgana.base_models import BaseVAE
from morgana.metrics import LF0Distortion
from morgana.sampling import UniformSphereSurfaceSampler
from morgana.viz.synthesis import MLPG
from morgana import utils

from misc import batch_synth, VAEExperimentBuilder

from tts_data_tools import data_sources


class _Encoder(nn.Module):
    def __init__(self, conditioning_dim, input_dim, dropout_prob, z_dim, latent=None):
        super(_Encoder, self).__init__()

        self.conditioning_dim = conditioning_dim
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.latent = latent

        self.shared_layer = utils.SequentialWithRecurrent(
            nn.Linear(self.conditioning_dim + self.input_dim, 256),
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
        )

        self.mu_layer = nn.Linear(64, self.z_dim)
        self.logvar_layer = nn.Linear(64, self.z_dim)

    def forward(self, inputs, seq_len=None):
        encoded = self.shared_layer(inputs, seq_len=seq_len)

        # Select the correct indices from the padded output.
        batch_idxs = torch.arange(encoded.shape[0], dtype=torch.long)
        encoded = encoded[batch_idxs, seq_len - 1, :]

        mean = self.mu_layer(encoded)
        log_variance = self.logvar_layer(encoded)

        return mean, log_variance


class VAE(BaseVAE):
    def __init__(self, z_dim=16, kld_weight=1., conditioning_dim=600+9, output_dim=1*3, dropout_prob=0., latent=None):
        """Initialises VAE parameters and settings."""
        super(VAE, self).__init__(z_dim=z_dim, kld_weight=kld_weight)
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.latent = latent

        self.encoder_layer = _Encoder(self.conditioning_dim, self.output_dim, dropout_prob, self.z_dim)

        self.decoder_layer = utils.SequentialWithRecurrent(
            nn.Linear(self.conditioning_dim + self.z_dim, 256),
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

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames'),
            'n_phones': data_sources.TextSource('n_phones'),
            'dur': data_sources.TextSource('dur', normalisation='mvn'),
            'lab': data_sources.NumpyBinarySource('lab', normalisation='minmax'),
            'counters': data_sources.NumpyBinarySource('counters', normalisation='minmax'),
            'lf0': data_sources.NumpyBinarySource('lf0', normalisation='mvn', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
        }

    def valid_data_sources(self):
        sources = self.train_data_sources()
        sources['sp'] = data_sources.NumpyBinarySource('sp')
        sources['ap'] = data_sources.NumpyBinarySource('ap')

        return sources

    def encode(self, features):
        # Prepare inputs.
        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(norm_lab, dur)

        norm_lf0_deltas = features['normalised_lf0_deltas']
        norm_counters = features['normalised_counters']
        encoder_inputs = torch.cat((norm_lf0_deltas, norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the encoder.
        n_frames = features['n_frames']
        mean, log_variance = self.encoder_layer(encoder_inputs, seq_len=n_frames)

        return mean, log_variance

    def decode(self, latent, features):
        # Prepare the inputs.
        n_frames = features['n_frames']
        max_n_frames = torch.max(n_frames)
        latents_at_frame_rate = latent.unsqueeze(1).repeat(1, max_n_frames, 1)

        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(norm_lab, dur)

        norm_counters = features['normalised_counters']
        decoder_inputs = torch.cat((latents_at_frame_rate, norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the decoder.
        pred_norm_lf0_deltas = self.decoder_layer(decoder_inputs, seq_len=n_frames)

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        pred_lf0 = MLPG(means=pred_lf0_deltas,
                        variances=self.normalisers['lf0'].delta_params['std_dev'] ** 2)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
            'lf0_deltas': pred_lf0_deltas,
            'lf0': pred_lf0
        }

        return outputs

    def predict(self, features):
        # Bypass this function, instead we will define the prediction within analysis_for_test_batch.
        pass

    def loss(self, input_features, output_features):
        inputs = input_features['normalised_lf0_deltas']
        outputs = output_features['normalised_lf0_deltas']
        seq_len = input_features['n_frames']

        latent = output_features['latent']
        mean = output_features['mean']
        log_variance = output_features['log_variance']

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(input_features['lf0'], output_features['lf0'], seq_len, input_features['vuv']))

        return self._loss(inputs, outputs, latent, mean, log_variance, seq_len)

    def analysis_for_valid_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        kwargs['sample_rate'] = sample_rate
        super(VAE, self).analysis_for_valid_batch(features, output_features, out_dir, **kwargs)
        batch_synth(features, output_features, out_dir, sample_rate)

    def analysis_for_test_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        kwargs['sample_rate'] = sample_rate
        batch_size = len(features['name'])

        # Oracle encoding as the latent.
        oracle_out_dir = os.path.join(out_dir, 'oracle')
        mean, _ = self.encode(features)
        oracle_output_features = self.decode(mean, features)
        super(VAE, self).analysis_for_test_batch(features, oracle_output_features, oracle_out_dir, **kwargs)

        # Zero vector as the latent.
        zeros_out_dir = os.path.join(out_dir, 'zeros')
        zeros = torch.zeros((batch_size, self.z_dim)).to(mean.device)
        zeros_output_features = self.decode(zeros, features)
        super(VAE, self).analysis_for_test_batch(features, zeros_output_features, zeros_out_dir, **kwargs)

        # For samples on the surface of a hypersphere as the latent.
        centre = torch.zeros(self.z_dim, device=mean.device)
        sphere_sampler = UniformSphereSurfaceSampler(centre, 3)
        for i in range(4):
            tail_out_dir = os.path.join(out_dir, 'tail_{}'.format(i))
            tail = sphere_sampler.sample([batch_size]).to(mean.device)
            tail_output_features = self.decode(tail, features)
            super(VAE, self).analysis_for_test_batch(features, tail_output_features, tail_out_dir, **kwargs)


def main():
    torch.random.manual_seed(1234567890)
    args = VAEExperimentBuilder.get_experiment_args()
    experiment = VAEExperimentBuilder(VAE, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

