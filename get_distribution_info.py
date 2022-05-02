import numpy as np
import argparse
import os
import sys
import csv


def get_label_info(args):
    label_dict=dict()   
    single_label_dict=dict()
    with open(args.file,encoding='latin1') as f:
        for (i, line) in enumerate(f):
            line = line.strip()
            items = line.split("\t")
            #keep labels list of string
            #line: text label username
            labels = (items[1].split(","))
            labels.sort()
            
            labelst=tuple(labels)
            if labelst not in label_dict:
                label_dict[labelst]=0
            label_dict[labelst]+=1
            if len(labels)==1:
                labels=labels[0]
                if labels not in single_label_dict:    
                    single_label_dict[labels]=[]
                single_label_dict[labels].append(items[0])

    return label_dict,single_label_dict

def create_files(args,single_label_dict):
    with open(args.out_file,'w',encoding='latin1') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in range(args.n):
            for l in single_label_dict:
                tsv_writer.writerow([single_label_dict[l][i]]+[l])





if __name__=='__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--file", type=str, required=True, help="file to analyze")
    cli_parser.add_argument("--n", type=int, required=False, help="how many lines per label to extract")
    cli_parser.add_argument("--out_file", type=str, required=False, default=None, help="output file containing extracted label lines")
    args = cli_parser.parse_args()

    label_dict,single_label_dict=get_label_info(args)
    print("There are {} labels: {}".format(len(label_dict.keys()),label_dict.keys()))
    for l in label_dict:
        print('label {} has {} examples'.format(l,label_dict[l]))
    if args.out_file is not None:
        create_files(args,single_label_dict)
        