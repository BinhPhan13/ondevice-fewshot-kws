cmd=(
python KWSFSL/test_fewshots.new.py
--data.cuda
--data.cuda_devices 2
--data.n_workers 16

--speech.default_datadir ~/MSWC/audio
--speech.csv_file mswc-subsets/en263.b1000.csv
--speech.use_wav
--speech.include_noise
--speech.noise_dir ~/MSWC/noise

--fsl.classifier ncm
--fsl.test.n_pos 5
--fsl.test.n_neg 50
--fsl.test.n_support 1
--fsl.test.threshold 1000
--fsl.test.batch_size 512
--fsl.test.n_episodes 1

--log.note en263
--model.model_path ~/results/HIC.1/best_model.pt
)
${cmd[@]}

