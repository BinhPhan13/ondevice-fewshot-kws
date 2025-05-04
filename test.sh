cmd=(
python KWSFSL/test_fewshots_classifiers_openset.py
--data.cuda
--data.cuda_devices 0

--speech.dataset googlespeechcommand
--speech.task GSC12,GSC22
--speech.default_datadir ~/GSC

--speech.include_unknown

--fsl.classifier ncm
--fsl.test.n_way 11
--fsl.test.n_support 10
--fsl.test.n_episodes 10
--fsl.test.batch_size 256

--model.model_path results/TL_MSWC500U_DSCNNLLN/best_model.pt
)
${cmd[@]}

