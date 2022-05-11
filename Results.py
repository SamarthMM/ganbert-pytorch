import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--file", type=str, required=True, help="ratio of unlabaled to labeled data")
cli_parser.add_argument("--reparse", type=bool, required=False, default=False, help="ratio of unlabaled to labeled data")
cli_parser.add_argument("--dataset", type=str, required=False, default='', help="twitter/goemotions")
args=cli_parser.parse_args()


if not os.path.exists(args.file[-5]+".json") or args.reparse:
    Experiments=[]
    n=100
    d=dict()
    with open(args.file,'r') as f:
        i=0
        for line in f:
            line=line.strip()
            if "Label ratio: " in line:
                Experiments.append(d)
                d=dict()
                d['epoch_info']= dict({
                    'num Epochs':0,
                    'av G loss train':[],
                    'av D loss train':[],
                    'Accuracy':[],
                    'Test Loss':[],
                })
                d['ratio']=float(line.split()[-1])
            elif "training file: " in line:
                d['num examples']=int(line.split('_')[-1][:-4])
            elif "======== Epoch" in line:
                d['epoch_info']['num Epochs']=int(line.split('/')[0].split()[-1]),
            elif "Average training loss generetor" in line:
                d['epoch_info']['av G loss train'].append(float(line.split()[-1]))
            elif "Average training loss discriminator" in line:
                d['epoch_info']['av D loss train'].append(float(line.split()[-1]))
            elif "Accuracy:" in line:
                d['epoch_info']['Accuracy'].append(float(line.split()[-1]))
            elif "Test Loss" in line:
                d['epoch_info']['Test Loss'].append(float(line.split()[-1]))
            #if(i>n):
            #    break
            i+=1

    Experiments.append(d)

    jsonf=json.dumps(Experiments)
    fj = open(args.file[-5]+".json","w")
    fj.write(jsonf)
    fj.close()


fignum=1
with open(args.file[-5]+".json",'r') as f:
    Experiments=json.load(f)
for exp in Experiments:
    if 'ratio' not in exp.keys():
        continue
    #print(exp.keys())
    #dict_keys(['epoch_info', 'epoch_info_temp', 'ratio', 'num examples'])
    ratio=exp['ratio']
    if 'num examples' not in exp.keys():
        nexamples=input("No num examples found. Enter it bc")
    else:
        nexamples=exp['num examples']
    n=exp['epoch_info']['num Epochs']
    if n==0:
        continue
    ln=list(range(n[0]))
    fig=plt.figure(fignum)
    fignum+=1
    plt.title('Average Generator Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(ln,exp['epoch_info']['av G loss train'])
    plt.savefig('Fig'+args.dataset+str(nexamples)+str(ratio)+'GenLoss.png')

    fig=plt.figure(fignum)
    fignum+=1
    plt.title('Average Discriminator Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(ln,exp['epoch_info']['av D loss train'])
    plt.savefig('Fig'+args.dataset+str(nexamples)+str(ratio)+'DiscLoss.png')

    fig=plt.figure(fignum)
    fignum+=1
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(ln,exp['epoch_info']['Accuracy'])
    plt.savefig('Fig'+args.dataset+str(nexamples)+str(ratio)+'Acc.png')

    fig=plt.figure(fignum)
    fignum+=1
    plt.title('Test Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.plot(ln,exp['epoch_info']['Test Loss'])
    plt.savefig('Fig'+args.dataset+str(nexamples)+str(ratio)+'TestLoss.png')

    plt.close('all')
    #accuracies=[exp['epoch_info'][i]['Accuracy'] for i in range(n)]
    #print(accuracies)




