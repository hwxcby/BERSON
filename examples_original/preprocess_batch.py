import torch
import numpy as np

def preprocess(batch):

    sen_num_dataset = []
    sample_len_dataset = []
    pairs_num_dataset = []

    for example in batch:
        sample_len_dataset.append(example['max_sample_len'])
        sen_num_dataset.append(example['passage_length'])
        pairs_num_dataset.append(example['pairs_num'])

    max_pair_len = max(sample_len_dataset)
    max_sen_num = max(sen_num_dataset)
    max_pairs_num = max(pairs_num_dataset)

    all_input_ids=[]
    all_attention_mask=[]
    all_token_type_ids=[]
    all_pairs_list=[]
    all_passage_length=[]
    all_pairs_num=[]
    all_sep_positions=[]
    all_ground_truth=[]
    all_mask_cls=[]
    all_pairwise_labels = []


    for inputs in batch:

        input_ids, masked_ids, token_type_ids, sep_positions = inputs['input_ids'], inputs['masked_ids'], inputs['token_type_ids'], inputs['sep_positions']
        shuffled_index, max_sample_len, ground_truth = inputs['shuffled_index'], inputs['max_sample_len'], inputs['ground_truth']
        passage_length, pairs_num, pairs_list = inputs['passage_length'], inputs['pairs_num'], inputs['pairs_list']
        pairwise_labels = inputs['pairwise_labels']


        padd_num_sen = max_sen_num - passage_length
        padding_pair_num = max_pairs_num - pairs_num
        pad_id = 0

        input_ids_new = []
        masked_ids_new = []
        token_type_ids_new = []
        pairwise_label_new = []

        for item in range(pairs_num): 
            padding_pair_len = max_pair_len - len(input_ids[item])

            input_ids_new.append(input_ids[item] + [pad_id] * padding_pair_len)
            masked_ids_new.append(masked_ids[item] + [pad_id] * padding_pair_len)
            token_type_ids_new.append(token_type_ids[item] + [pad_id] * padding_pair_len)

        ### padding for padded pairs
        input_ids_new = input_ids_new + [[pad_id] * max_pair_len] * padding_pair_num
        masked_ids_new = masked_ids_new + [[pad_id] * max_pair_len] * padding_pair_num   
        token_type_ids_new = token_type_ids_new + [[pad_id] * max_pair_len] * padding_pair_num
        pairwise_labels_new = pairwise_labels + [0] * padding_pair_num 


        pairs_list_new = pairs_list + [[0,1]] * padding_pair_num
        passage_length_new = passage_length
        pairs_num_new = pairs_num
        sep_positions_new = sep_positions + [[2,6]] * padding_pair_num

        mask_cls_new = [1] * passage_length_new + [pad_id] * padd_num_sen
        ground_truth_new = ground_truth + [pad_id] * padd_num_sen


        all_input_ids.append(input_ids_new)
        all_attention_mask.append(masked_ids_new)
        all_token_type_ids.append(token_type_ids_new)
        all_pairs_list.append(pairs_list_new)
        all_passage_length.append(passage_length_new)
        all_pairs_num.append(pairs_num_new)
        all_sep_positions.append(sep_positions_new)
        all_ground_truth.append(ground_truth_new)
        all_mask_cls.append(mask_cls_new)
        all_pairwise_labels.append(pairwise_labels_new)


    all_input_ids=torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask=torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids=torch.tensor(all_token_type_ids, dtype=torch.long)
    all_pairs_list=torch.tensor(all_pairs_list, dtype=torch.long)
    all_passage_length=torch.tensor(all_passage_length, dtype=torch.long)
    all_pairs_num=torch.tensor(all_pairs_num, dtype=torch.long)
    all_sep_positions=torch.tensor(all_sep_positions, dtype=torch.long)
    all_ground_truth=torch.tensor(all_ground_truth, dtype=torch.long)
    all_mask_cls=torch.tensor(all_mask_cls, dtype=torch.long)
    all_pairwise_labels=torch.tensor(all_pairwise_labels, dtype=torch.long)

    new_batch=[all_input_ids, all_attention_mask, all_token_type_ids, all_pairs_list, all_passage_length, all_pairs_num, all_sep_positions, all_ground_truth, all_mask_cls, all_pairwise_labels]

    return new_batch





