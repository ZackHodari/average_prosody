import os

import numpy as np
from scipy.signal import savgol_filter
import torch
from tqdm import tqdm

from morgana.experiment_builder import ExperimentBuilder
from morgana import lr_schedules
from morgana import utils
from morgana import _logging
from tts_data_tools import file_io
from tts_data_tools.wav_gen import world


def batch_synth(features, output_features, out_dir, sample_rate=16000):
    synth_dir = os.path.join(out_dir, 'synth')
    os.makedirs(synth_dir, exist_ok=True)

    lf0, vuv, sp, ap = utils.detach_batched_seqs(
        output_features['lf0'], features['vuv'], features['sp'], features['ap'],
        seq_len=features['n_frames'])

    for i, name in enumerate(features['name']):
        f0_i = np.exp(lf0[i])
        f0_i = savgol_filter(f0_i, 7, 1)

        wav_path = os.path.join(synth_dir, '{}.wav'.format(name))
        wav = world.synthesis(f0_i, vuv[i], sp[i], ap[i], sample_rate=sample_rate)
        file_io.save_wav(wav, wav_path, sample_rate=sample_rate)


class VAEExperimentBuilder(ExperimentBuilder):

    @classmethod
    def add_args(cls, parser):
        super(VAEExperimentBuilder, cls).add_args(parser)

        parser.add_argument("--kld_wait_epochs",
                            dest="kld_wait_epochs", action="store", type=int, default=5,
                            help="Number of epochs to wait with the KLD cost at 0.0")
        parser.add_argument("--kld_warmup_epochs",
                            dest="kld_warmup_epochs", action="store", type=int, default=20,
                            help="Number of epochs to increase the KLD cost from 0.0, to avoid posterior collapse.")

    def __init__(self, model_class, experiment_name, **kwargs):
        self.kld_wait_epochs = kwargs['kld_wait_epochs']
        self.kld_warmup_epochs = kwargs['kld_warmup_epochs']

        super(VAEExperimentBuilder, self).__init__(model_class, experiment_name, **kwargs)

    def train_epoch(self, data_generator, optimizer, lr_schedule=None, gen_output=False, out_dir=None):
        self.model.mode = 'train'
        self.model.metrics.reset_state('train')

        loss = 0.0
        pbar = _logging.ProgressBar(len(data_generator))
        for i, features in zip(pbar, data_generator):
            self.model.step = (self.epoch - 1) * len(data_generator) + i + 1

            # Anneal the KL divergence, linearly increasing from 0.0 to the initial KLD weight set in the model.
            if self.kld_wait_epochs != 0 and self.epoch == self.kld_wait_epochs + 1 and self.kld_warmup_epochs == 0:
                self.model.kld_weight = self.model.max_kld_weight
            if self.kld_warmup_epochs != 0 and self.epoch > self.kld_wait_epochs:
                if self.model.kld_weight < self.model.max_kld_weight:
                    self.model.kld_weight += self.model.max_kld_weight / (self.kld_warmup_epochs * len(data_generator))
                    self.model.kld_weight = min(self.model.max_kld_weight, self.model.kld_weight)

            self.model.tensorboard.add_scalar('kl_weight', self.model.kld_weight, global_step=self.model.step)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            batch_loss, output_features = self.model(features)

            batch_loss.backward()
            optimizer.step()

            # Update the learning rate.
            if lr_schedule is not None and self.lr_schedule_name in lr_schedules.BATCH_LR_SCHEDULES:
                lr_schedule.step()

            loss += batch_loss.item()

            # Update the exponential moving average model if it exists.
            if self.ema_decay:
                self.ema.update_params(self.model)

            # Log metrics.
            pbar.print('train', self.epoch,
                       kld_weight=tqdm.format_num(self.model.kld_weight),
                       batch_loss=tqdm.format_num(batch_loss),
                       **self.model.metrics.results_as_str_dict('train'))

            if gen_output:
                self.model.analysis_for_train_batch(features, output_features,
                                                    out_dir=out_dir, sample_rate=self.sample_rate)

        if gen_output:
            self.model.analysis_for_train_epoch(out_dir=out_dir, sample_rate=self.sample_rate)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            file_io.save_json(self.model.metrics.results_as_json_dict('train'),
                              os.path.join(out_dir, 'metrics.json'))

        self.model.mode = ''

        return loss / (i + 1)

    def run_train(self):
        if self.kld_wait_epochs > 0 or self.kld_warmup_epochs != 0:
            self.model.max_kld_weight = self.model.kld_weight
            self.model.kld_weight = 0.0
        super(VAEExperimentBuilder, self).run_train()


class GaussianMixtureModel(torch.distributions.Distribution):
    def __init__(self, pi, mu, sigma):
        # [batch_size, seq_len, n_components]
        self.pi = pi
        # [batch_size, seq_len, n_components, feat_dim]
        self.mu = torch.stack(mu, dim=2)
        # [batch_size, seq_len, n_components, feat_dim]
        self.sigma = torch.stack(sigma, dim=2)

        self.n_components = self.pi.shape[-1]

        self.component_dist = torch.distributions.Categorical(probs=pi)
        self.mixture_dists = [torch.distributions.Normal(loc=mu[i], scale=sigma[i])
                              for i in range(self.n_components)]

        batch_shape = self.pi.size()[:-1] if self.pi.ndimension() > 1 else torch.Size()
        super(GaussianMixtureModel, self).__init__(batch_shape)

    @classmethod
    def create_component_indices(cls, k):
        batch_size, max_seq_len = k.shape[:2]

        # Using the number of batch and sequence items, create an index array so we can index with `k`.
        # All three of `batch_idxs`, `seq_idxs`, and `k` have shape [batch_size, max_seq_len].
        batch_idxs = torch.arange(batch_size)[:, None].repeat(1, max_seq_len).to(k.device)
        seq_idxs = torch.arange(max_seq_len)[None, :].repeat(batch_size, 1).to(k.device)

        # As we want each item in `k` to index a different batch and time index we need to use three index values for
        # each item in `k`, therefore the shape of this indexing array is [3, batch_size, max_seq_len].
        dist_idxs = torch.stack((batch_idxs, seq_idxs, k))

        return tuple(dist_idxs)

    def argmax_components(self):
        k = self.pi.argmax(dim=2)
        idxs = GaussianMixtureModel.create_component_indices(k)

        return self.mu[idxs], self.sigma[idxs]

    def sample(self, sample_shape=torch.Size()):
        k = self.component_dist.sample(sample_shape).type(torch.long)
        dist_idxs = self.create_component_indices(k)

        dist = torch.distributions.Normal(loc=self.mu[dist_idxs], scale=self.sigma[dist_idxs])
        return dist.sample()

    def log_prob(self, value):
        mixture_log_likelihoods = [dist.log_prob(value) for dist in self.mixture_dists]
        mixture_log_likelihoods = torch.stack(mixture_log_likelihoods, dim=2)
        log_pi = torch.log(self.pi)[:, :, :, None]

        # Perform a dot product over the third axis using sum-product, shapes: [B, T, N] (dot) [B, T, N, F] = [B, T, F].
        log_likelihood = torch.sum(log_pi + mixture_log_likelihoods, dim=2)
        return log_likelihood


