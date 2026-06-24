cd /root/autodl-tmp/opd_mopd/OPD-code

screen -dmS mopd_tensorboard bash -lc '
cd /root/autodl-tmp/opd_mopd/OPD-code &&
tensorboard --logdir tensorboard_log --host 127.0.0.1 --port 6006
'