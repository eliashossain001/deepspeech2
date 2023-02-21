import kenlm
import torch
from collections import OrderedDict
from data_utils import TextTransform
from pyctcdecode import build_ctcdecoder
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import pandas as pd


def beam_search_decoder(data, beam_width, n_best):
   
    df = pd.DataFrame(data)
    data = tf.nn.softmax(df.values)
    print('Data:', data)

    sequences = [[list(), [], 0.0]]

    results = []
    for i, row in enumerate(data):
        all_candidates = list()
        for seq, seq_log_probs, score in sequences:
            for j in range(len(row)):
                candidate_seq = seq + [j]
                candidate_seq_log_probs = seq_log_probs + [np.log(row[j])]
                # compute score
                next_token_log_prob = np.log(row[j])
                candidate_score = sum(candidate_seq_log_probs) / len(candidate_seq_log_probs) + next_token_log_prob * len(candidate_seq)
                #print('candidate score:', candidate_score)
                candidate = [candidate_seq, candidate_seq_log_probs, candidate_score]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[2], reverse=True)
        sequences = ordered[:beam_width]
        n_best_sequences = sequences[:n_best]
        results.append(n_best_sequences)
    
        # Create a dictionary to map the token indices to their corresponding phoneme names
        token_index_to_name = dict(enumerate(range(len(row))))
        index_to_phoneme = {v: k for k, v in token_index_to_name.items()}

        # Convert token indices back to their corresponding phoneme names
        final_results = []
        for step_results in results:
            step_token_scores = []
            for seq, seq_log_probs, score in step_results:
                # Convert token indices to their corresponding phoneme names
                #token_seq = [index_to_phoneme[i] for i in seq]
                token_seq= [index_to_phoneme[i] for i in seq]
                step_token_scores.append((token_seq[-1], score))
                #step_token_scores.append((token_seq[-1], score if not np.isnan(score) else -1e9))

            final_results.append(step_token_scores)
                    
    return final_results


# if __name__ == "__main__":
#     #print(refine_output('t̪t̪ʰt̪'))