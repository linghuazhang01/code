cd /root/autodl-tmp/opd_mopd/OPD-code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

screen -dmS mopd_tensorboard bash -lc '
cd /root/autodl-tmp/opd_mopd/OPD-code &&
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate mopd-verl &&
tensorboard --logdir tensorboard_log --host 127.0.0.1 --port 6006
'