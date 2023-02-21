import kenlm
import torch
from collections import OrderedDict
from data_utils import TextTransform
from pyctcdecode import build_ctcdecoder
import numpy as np
import tensorflow as tf
def refine_output(phoneme_sequence: str):
    # base phoneme
    jukto_vocab = ['ã', 'bʰ', 'cʰ', 'dʰ', 'd̪', 'd̪ʰ', 'ẽ', 'gʰ','ĩ', 'i̯', 'kʰ', 
         'õ', 'o̯', 'pʰ', 'tʰ', 't̪', 't̪ʰ', 'ũ', 'u̯', 'æ̃', 'ɔ̃', 'ɟʰ', 'ɽʰ']
    
    
    result = {}
        
    for j_v in jukto_vocab:
        for j,_ in enumerate(phoneme_sequence):
            if phoneme_sequence[j:j + len(j_v)] == j_v:
                if j not in result.keys():
                    result[j] = len(j_v)
                else:
                    if len(j_v) > result[j]:
                        result[j] = len(j_v)


    result = OrderedDict(sorted(result.items()))
    output = ""
    ind = 0
    for key in result.keys():
        while(1):
            if ind < key:
                output += phoneme_sequence[ind] + ','
                ind += 1
            else:
                output += phoneme_sequence[ind:ind+result[key]]+','
                ind += result[key]
                break

    
    if len(phoneme_sequence) > ind:
        temp = int(len(phoneme_sequence) - ind)
        for i in range(0, temp):
            output += phoneme_sequence[ind] +','
            ind += 1        
    
    return output



def GreedyDecoder(output,text_transform, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)                                                                                                   
    decodes = []
    for i, args in enumerate(arg_maxes):
        # print(args)
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:                        
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes





def BeamSearch(output, top_n = 5):
    # base phoneme 
    vocab = ['_','a', 'ã', 'b', 'bʰ', 'c', 'cʰ', 'd', 'dʰ', 'd̪', 'd̪ʰ', 'e', 'ẽ', 'g', 'gʰ', 'h', 'i', 'ĩ', 'i̯', 'k', 'kʰ', 
             'l', 'm', 'n', 'o', 'õ', 'o̯', 'p', 'pʰ', 'r', 's', 't', 'tʰ', 't̪', 't̪ʰ', 'u', 'ũ', 'u̯', 'æ', 'æ̃', 'ŋ', 
             'ɔ', 'ɔ̃', 'ɟ', 'ɟʰ', 'ɽ', 'ɽʰ', 'ʃ', 'ʲ', 'ʷ', '@']
    
    decoder = build_ctcdecoder(
        vocab
        
    )
    #print('decoder:', decoder)

    out = decoder.decode_beams(output)
    
    output = []
    for i in range(top_n):
        comma_seperated_phoneme_sequence = refine_output(out[i][0])
        #print('candidate sequence:',comma_seperated_phoneme_sequence)
        output.append((str(comma_seperated_phoneme_sequence), out[i][4]))
        #print('Outut is',output)
    return output




def per(ref, hyp ,debug=True):
    ref_b = refine_output(ref)
    # print("ref_b: ",ref_b)
    hyp_b = refine_output(hyp)
    # print("hyp_b: ",hyp_b)

    r = ref_b.split(",")
    h = hyp_b.split(",")

    #r = ref_b.split("@")
    #h = hyp_b.split("@")

    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\t\t\tREF\t\t\tHYP")
        with open('debug.txt','w') as f:
            f.write("OP\t\t\tREF\t\t\tHYP\n")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "**" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
    
                lines.append("DEL\t" + r[i]+"\t"+"**")
               
    if debug:
        lines = reversed(lines)
        count_ln = 0
        for line in lines:

            #count = 0  
            print(str(count_ln)+':    '+line)
            with open('debug.txt','a') as f:
                f.write(str(count_ln)+':    '+line+"\n")
            count_ln = count_ln + 1

        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
        
        # with open('debug.txt','w') as f:
        #     f.write("#cor " + str(numCor)+'\n')
        #     f.write("#sub " + str(numSub)+'\n')
        #     f.write("#del " + str(numDel)+'\n')
        #     f.write("#ins " + str(numIns)+'\n')
    
    # return (numSub + numDel + numIns) / (float) (len(r))
    per_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    #output_per = print(per_result)
    return {'PER':per_result, 'numCor':numCor, 'numSub':numSub, 'numIns':numIns, 'numDel':numDel, "numCount": len(r)}


if __name__ == "__main__":
    print(refine_output('t̪t̪ʰt̪'))