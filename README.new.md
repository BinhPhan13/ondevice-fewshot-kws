# Few-Shot Open-Set Learning for On-Device Customization of KWS
Fork from the [original repo](https://github.com/mrusci/ondevice-learning-kws) with some optimizations

## MSWC Dataset
* The audio files used for the dataset is specified in --speech.csv_file
* To sample this csv file, used the mswc-sample and mswc-sgo.
* Specified the sample options with mswc-sample (-h for help) to preview, then run mswc-sgo output_file + the options


## Training
Run the script train.sh, some important configurations:
* --model.encoding: name of the encoder, available list specified in KWSFSL/models/repr_model.py
* --train.(n_way, n_support, n_query, n_episodes, n_epochs)
* --log.exp_dir: the folder to save checkpoints and the model

## Evaluation
Run the script test.new.sh, some important configurations:
* --model.model_path: the path to the model (usually the the folder specified in --log.exp_dir / best_model.pt)
* --fsl.test.(n_pos, n_neg, n_support, threshold, n_episodes)

The evaluation results are saved in the same folder of the --model.model_path.<br>
Refer to DET.ipynb for DET curve visualization between models.

