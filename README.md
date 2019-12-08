# average_prosody
Code for paper titled "Using generative modelling to produce varied intonation for speech synthesis" submitted to the
Speech Synthesis Workshop.

Speech samples are available at
[zackhodari.github.io/SSW_2019_average_prosody](http://zackhodari.github.io/SSW_2019_average_prosody.html).


The models defined in this repo should be run using the [`Morgana`](https://github.com/ZackHodari/morgana) toolkit.



# Data setup
Extraction with [`tts_data_tools`](https://github.com/ZackHodari/tts_data_tools) currently requires wavfiles at the
desired frame-rate (16kHz in the paper), and label files with time alignments (label_state_align).

If you want to prepare your own data to use with [`Morgana`](https://github.com/ZackHodari/morgana) you will need a
directory for each data split (e.g. `train`). Each of these must contain individual directories for each feature being
loaded, this will then be loaded by the `morgana.data._DataSource` instances created within the model file, you can
write your own data sources to load any file type you need. See
[`Morgana`'s data loading](https://zackhodari.github.io/morgana/reference/morgana.data.html#morgana.data._DataSource)
for more details.

```bash
pip install git+https://github.com/zackhodari/tts_data_tools

cd ~/data/Blizzard2017

tdt_process_dataset \
    --lab_dir label_state_align \
    --wav_dir wav_16000 \
    --out_dir train \
    --question_file questions-unilex_dnn_600.hed \
    --id_list train_file_id_list.scp \
    --state_level \
    --subphone_feat_type full \
    --calculate_normalisation \
    --normalisation_of_deltas

tdt_process_dataset \
    --lab_dir label_state_align \
    --wav_dir wav_16000 \
    --out_dir valid \
    --question_file questions-unilex_dnn_600.hed \
    --id_list valid_file_id_list.scp \
    --state_level \
    --subphone_feat_type full
```



# Usage

```bash
pip install git+https://github.com/zackhodari/morgana

git clone https://github.com/ZackHodari/average_prosody.git
cd average_prosody
mkdir experiments

python f0_RNN.py --experiment_name RNN \
    --data_root ~/data/Blizzard2017 \
    --train_dir train --train_id_list train_file_id_list.scp \
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --end_epoch 100 \
    --num_data_threads 4 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --lr_schedule_name noam \
    --lr_schedule_kwargs "{'warmup_steps': 1000}"

python f0_RNN_scaled.py --experiment_name RNN_scaled \
    --data_root ~/data/Blizzard2017 \
    --no-train \
    --checkpoint_path experiments/RNN/checkpoints/epoch_30.pt
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --num_data_threads 4 \
    --batch_size 32

python f0_MDN.py --experiment_name MDN \
    --data_root ~/data/Blizzard2017 \
    --train_dir train --train_id_list train_file_id_list.scp \
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --end_epoch 100 \
    --num_data_threads 4 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --lr_schedule_name noam \
    --lr_schedule_kwargs "{'warmup_steps': 1000}" \
    --model_kwargs "{'var_floor': 1e-4}"

python f0_VAE.py --experiment_name VAE \
    --data_root ~/data/Blizzard2017 \
    --train_dir train --train_id_list train_file_id_list.scp \
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --test_dir valid --test_id_list valid_file_id_list.scp \
    --end_epoch 100 \
    --num_data_threads 4 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --lr_schedule_name noam \
    --lr_schedule_kwargs "{'warmup_steps': 1600}" \
    --kld_wait_epochs 1 \
    --kld_warmup_epochs 40 \
    --model_kwargs "{'kld_weight': 0.01}"
```

