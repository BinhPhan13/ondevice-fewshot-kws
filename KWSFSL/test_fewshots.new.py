import os
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from utils import filter_opt


def test_model(classifier, test_loader):
    results = None
    for sample in tqdm(test_loader):
        x = sample['data']
        labels = sample['label']
        scores, target_ids = classifier.evaluate_batch(
            x, labels, return_probas=False
        )
        batch_results = torch.cat((target_ids.view(-1, 1), scores), 1)

        if results is None:
            results = batch_results
        else:
            results = torch.cat((results, batch_results), 0)

    return results


if __name__ == '__main__':
    from parser_kws import parser
    args = parser.parse_args()
    opt = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['data.cuda_devices']

    # load the encoder
    model_path = Path(opt['model.model_path'])
    if model_path.is_file():
        enc_model = torch.load(model_path)
    else:
        raise ValueError(f"Model {model_path} not valid")

    # load the classifier
    import classifiers
    _name_mapping = {
        'ncm': 'NearestClassMean',
        'ncm_openmax': 'NCMOpenMax',
        'peeler': 'PeelerClass',
        'dproto': 'DProto',
    }
    classifier_name = opt['fsl.classifier']
    if classifier_name not in _name_mapping:
        raise ValueError(f'{classifier_name} not available')

    logger.info(f'Using the classifier: {classifier_name}')
    classifier = getattr(classifiers, _name_mapping[classifier_name])(
        backbone=enc_model, cuda=opt['data.cuda']
    )

    from data.mswc import MSWCDataset
    speech_args = filter_opt(opt, 'speech')
    dataset = MSWCDataset(speech_args)
    test_args = filter_opt(opt, 'fsl.test')

    npos = test_args['n_pos']
    nneg = test_args['n_neg']
    nshot = test_args['n_support']
    threshold = test_args['threshold']

    get_eval_dataloaders = partial(
        dataset.get_eval_dataloaders,
        npos=npos,
        nneg=nneg,
        nshot=nshot,
        threshold=threshold,
        batch_size=test_args['batch_size'],
        nworkers=opt['data.nworkers'],
    )

    n_episodes = test_args['n_episodes']
    note = opt['log.note']
    log_dir = model_path.parent / (
        f"eval_fsl.{note}.{classifier_name}."
        f"{npos}pos.{nneg}neg.{nshot}shot.{n_episodes}eps"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        dl_train, dl_test = get_eval_dataloaders()

        support_sample = next(iter(dl_train))
        support_samples = support_sample['data']
        class_list = support_sample['label'][0]

        logger.info(f'Test Episode {ep} with classes: {class_list}')

        classifier.fit_batch_offline(support_samples, class_list)
        results = test_model(classifier, dl_test)

        torch.save(results.cpu(), log_dir/f'ep{ep}.pt')

