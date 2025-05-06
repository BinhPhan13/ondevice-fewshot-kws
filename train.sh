cmd=(
python KWSFSL/metric_learning.py
--data.cuda
--data.cuda_devices 0
--data.n_workers 16

--speech.dataset MSWC
--speech.task MSWC500U
--speech.default_datadir ~/MSWC/audio
--speech.csv_file ~/mswc-subsets/en500.orig.csv

--speech.use_wav
--speech.include_noise
--speech.noise_dir ~/MSWC/noise

--model.model_name repr_conv
--model.encoding DSCNNL_LAYERNORM
--model.z_norm

--train.loss triplet
--train.margin 0.5

--train.n_way 80
--train.n_support 0
--train.n_query 20
--train.n_episodes 400
--train.epochs 40

--log.exp_dir ~/results/HIC.1
)
${cmd[@]}

