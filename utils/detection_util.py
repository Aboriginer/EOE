import os
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as sk
from transformers import CLIPTokenizer, AutoTokenizer, BertTokenizer
import torch.nn.functional as F
from .imagenet_templates import openai_imagenet_template_subset


def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_ood_scores_clip(args, net, loader, test_labels, gpt_labels, softmax = True):
    net.eval()
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    id_class_nums = len(test_labels)
    if args.model == 'CLIP':
        tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)
    elif args.model == 'ALIGN':
        tokenizer = BertTokenizer.from_pretrained("kakaobrain/align-base")
    elif args.model == 'GroupViT':
        tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
    elif args.model == 'AltCLIP':
        tokenizer = AutoTokenizer.from_pretrained("BAAI/AltCLIP")

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            labels = labels.long().cuda()
            images = images.cuda()
            
            image_features = net.get_image_features(pixel_values = images).float() # 500x512
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if args.score in ['MCM', 'energy', 'max-logit']:
                if not args.ensemble:
                    text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
                    text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                                    attention_mask = text_inputs['attention_mask'].cuda()).float()  
                else:
                    text_features = clip_text_ens(net, tokenizer, test_labels)
                text_features /= text_features.norm(dim=-1, keepdim=True) # cls x 512
                output = image_features @ text_features.T # 500 x cls
            elif args.score == 'EOE':
                # Since the near prompt generates candidate OOD class names based on each ID class name, some candidate OOD class names might overlap with other ID class names
                if args.ood_task == 'near':
                    gpt_labels = remove_overlap_class(test_labels, gpt_labels)
                total_features = pre_filter(net, tokenizer, test_labels, gpt_labels, args)
                total_features /= total_features.norm(dim=-1, keepdim=True)
                output = image_features @ total_features.T
            
            if softmax:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            else:
                smax = to_np(output/ args.T)
            
            # cal score
            if args.score == 'EOE':
                if args.score_ablation == 'EOE':
                    smax = np.max(smax[:, :id_class_nums], axis=1) - args.beta * np.max(smax[:, id_class_nums:], axis=1)
                    _score.append(-smax)
                elif args.score_ablation == 'MAX':
                    iid_values, iid_indices = torch.max(output[:, :id_class_nums], dim=1)
                    ood_values, ood_indices = torch.max(output[:, id_class_nums:], dim=1)
                    condition = ood_values > iid_values
                    output[:, :id_class_nums][condition] = 1/id_class_nums
                    output = output[:, :id_class_nums]
                    smax = to_np(F.softmax(output/ args.T, dim=1))
                    smax = np.max(smax[:, :id_class_nums], axis=1)
                    _score.append(-smax)
                elif args.score_ablation == 'MSP':
                    smax = np.max(smax[:, :id_class_nums], axis=1)
                    _score.append(-smax)
                elif args.score_ablation == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output[:, :id_class_nums] / args.T, dim=1)) - (args.T*torch.logsumexp(output[:, id_class_nums:] / args.T, dim=1))))
                elif args.score_ablation == 'max-logit':
                    _score.append(-to_np(torch.max(output[:, :id_class_nums], 1)[0] - torch.max(output[:, id_class_nums:], 1)[0]))
                else:
                    raise NotImplementedError
            elif args.score == 'MCM':
                _score.append(-np.max(smax, axis=1)) 
            elif args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'max-logit':
                _score.append(-to_np(torch.max(output, 1)[0]))

    return concat(_score)[:len(loader.dataset)].copy()  


def remove_overlap_class(test_labels, gpt_labels):
    words_set = {word for phrase in test_labels for word in phrase.split()}
    gpt_labels = [phrase for phrase in gpt_labels if not any(word in words_set for word in phrase.split())]

    return gpt_labels


def clip_text_ens(net, tokenizer, test_labels):
    net.eval()
    prompt_pool = openai_imagenet_template_subset[0]
    all_prompts = [template(label) for label in test_labels for template in prompt_pool]
    text_inputs = tokenizer(all_prompts, padding=True, return_tensors="pt")
    all_encoded_features = net.get_text_features(input_ids=text_inputs['input_ids'].cuda(), 
                                                 attention_mask=text_inputs['attention_mask'].cuda()).float()

    all_text_features = torch.zeros(len(test_labels), all_encoded_features.shape[1]).to(all_encoded_features)
    for i in range(len(test_labels)):
        for j in range(len(prompt_pool)):
            index = i * len(prompt_pool) + j

            tmp = all_encoded_features[index] / all_encoded_features[index].norm(dim=-1, keepdim=True)
            all_text_features[i] += tmp

    return all_text_features


def pre_filter(net, tokenizer, test_labels, gpt_labels, args):
    # Filtering using cosine similarity
    net.eval()
    if not args.ensemble:
        test_labels_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
        test_labels_features = net.get_text_features(input_ids = test_labels_inputs['input_ids'].cuda(), 
                                                    attention_mask = test_labels_inputs['attention_mask'].cuda()).float()
        
        gpt_labels_inputs = tokenizer([f"a photo of a {c}" for c in gpt_labels], padding=True, return_tensors="pt")
        gpt_labels_features = net.get_text_features(input_ids = gpt_labels_inputs['input_ids'].cuda(), 
                                                    attention_mask = gpt_labels_inputs['attention_mask'].cuda()).float()
    else:
        test_labels_features = clip_text_ens(net, tokenizer, test_labels)
        gpt_labels_features = clip_text_ens(net, tokenizer, gpt_labels)


    cosine_sim = torch.empty((gpt_labels_features.shape[0], test_labels_features.shape[0]))
    for i in range(gpt_labels_features.shape[0]):
        cosine_sim[i] = F.cosine_similarity(gpt_labels_features[i].unsqueeze(0), test_labels_features, dim=1)

    #  prevent excessive ID samples from being classified as OOD candidates
    if args.ood_task == 'near' and args.in_dataset == 'ImageNet20':
        threshold = 0.85
    else:
        threshold = 1.0
    mask = (cosine_sim > threshold).any(dim=1)

    if threshold >= 1:
        # print(cosine_sim[cosine_sim > 1])
        assert torch.all(~mask), "pre_filter: error in mask"

    gpt_labels_features_filtered = gpt_labels_features[~mask]
    
    return torch.cat((test_labels_features, gpt_labels_features_filtered), dim=0)


def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)
