import os
os.environ['CURL_CA_BUNDLE'] = '' # for SSLError: HTTPSConnectionPool
import argparse
import numpy as np
import torch
from scipy import stats
from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_val_loader, set_ood_loader_ImageNet
from utils.generate_llm_class import load_llm_classes
from utils.args_pool import *


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset.startswith("ImageNet_C"):
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']
    elif args.in_dataset == 'ImageNet':
        if args.ood_task.startswith("far"):
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']
        else:
            out_datasets = ['ssb_hard', 'ninco']
    else:
        out_datasets = dataset_mappings.get(args.in_dataset, [])


    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)

    test_labels = list(test_labels)
    if args.score == 'EOE':
        llm_labels = load_llm_classes(args, test_labels)
    else:
        llm_labels = []

    print(f"test label: {test_labels}")
    print(f"gpt label: {llm_labels}")

    in_score  = get_ood_scores_clip(args, net, test_loader, test_labels, llm_labels)
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
        out_score = get_ood_scores_clip(args, net, ood_loader, test_labels, llm_labels)
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


def process_args():
    parser = argparse.ArgumentParser(description='Leverage LLMs for OOD Detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='bird200', type=str, choices=ALL_ID_DATASET, help='in-distribution dataset')
    parser.add_argument('--root_dir', default="/data2/chentao/code/DOS/datasets", type=str, help='root dir of datasets')
    # prompt pipeline
    parser.add_argument('--ensemble', action='store_true', default=False, help='CLIP text prompt engineering')
    parser.add_argument('--L', type=int, default=500, help='the length of envisioned OOD class labels, for far/fine-grained: L=500, for near: L=3')
    parser.add_argument('--beta', type=float, default=0.25, help='beta in Eq. 3')
    parser.add_argument('--ood_task', type=str, default='far', choices=ALL_OOD_TASK, help='choose ood task')
    parser.add_argument('--generate_class', action='store_true', help='whether to envision OOD candidate classes or loaded from existing JSONs')
    parser.add_argument('--json_number', type=int, default=0, help='loaded json number')
    parser.add_argument('--llm_model', default="gpt-3.5-turbo-16k", type=str, choices=ALL_LLM, help='LLMs')
    parser.add_argument('--name', default="eval_ood", type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int, help='the GPU indice to use')
    parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--T', type=float, default=1, help='score temperature parameter') # It is better to set T to 0.01 for energy score in our framework
    parser.add_argument('--model', default='CLIP', type=str, choices=['CLIP', 'ALIGN', 'GroupViT', 'AltCLIP'], help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16', choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['EOE', 'MCM', 'energy', 'max-logit'], help='args.score is for different comparison methods')

    parser.add_argument('--score_ablation', default='MAX', type=str, choices=['MAX', 'MSP', 'energy', 'max-logit', 'EOE'], help='args.score_ablation is for ablation studies in Score Functions (Sec. 4.3), i.e., the score function below will use the candidate OOD class names')
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim, 512 for ViT-B and 768 for ViT-L')
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    
    os.makedirs(args.log_directory, exist_ok=True)

    return args


if __name__ == '__main__':
    main()