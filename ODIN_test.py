"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import numpy as np
import calculate_log as callog
import os
import lib_generation
from pathlib import Path

def test_ODIN(model, test_loader, out_test_loader, net_type, args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outf = "./output/"+str(args.gpu)
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/dog/'+ net_type + '.pth.tar'
    save_fig = "./save_fig/dog"
    outf = outf + net_type
    Path(outf).mkdir(exist_ok = True, parents=True)
    Path(save_fig).mkdir(exist_ok=True, parents=True)
    torch.cuda.manual_seed(0)
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    # check the in-distribution dataset
    model = model.to(device).eval()

    # measure the performance
    M_list = [0.001, 0.0014, 0.002]
    T_list = [100]
    # M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    # T_list = [1, 10, 100, 1000]
    base_line_list = []
    ODIN_best_tnr = [0]                 #### OOD dataset이 여러개면 [0,0,0]...
    ODIN_best_results = [0]
    ODIN_best_temperature = [-1]
    ODIN_best_magnitude = [-1]

    f = open(os.path.join(save_fig, "save_result.txt"), "w")

    init_txt = ""
    for mtype in mtypes:
        init_txt += ' {mtype:6s}'.format(mtype=mtype)
    f.write(init_txt)

    lib_generation.get_posterior(model, net_type, test_loader, 0, 1, outf, True)
    lib_generation.get_posterior(model, net_type, out_test_loader, 0, 1, outf, False)
    ODIN_results = callog.metric(outf,save_fig, 1, 0, ['PoT'])
    result = "\n{:6.2f}{:6.2f}{:6.2f}{:6.2f}{:6.2f}".format(100. * ODIN_results["PoT"]["TNR"],
                                                            100. * ODIN_results["PoT"]["AUROC"],
                                                            100. * ODIN_results["PoT"]["DTACC"],
                                                            100. * ODIN_results["PoT"]["AUIN"],
                                                            100. * ODIN_results["PoT"]["AUOUT"],
                                                            )
    f.write(result)
    out_count = 0
    ODIN_best_results[out_count] = ODIN_results
    ODIN_best_tnr[out_count] = ODIN_results['PoT']['TNR']
    ODIN_best_temperature[out_count] = 1
    ODIN_best_magnitude[out_count] = 0

    for T in T_list:
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, net_type, test_loader, magnitude, temperature, outf, True)
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude))



            # print('Out-distribution: ' + "OOD")
            lib_generation.get_posterior(model, net_type, out_test_loader, magnitude, temperature, outf, False)
            if temperature == 1 and magnitude == 0:
                test_results = callog.metric(outf,save_fig, stypes=['PoT'])
                base_line_list.append(test_results)
            else:
                ODIN_results = callog.metric(outf,save_fig, temperature, magnitude, ['PoT'])
                result = "\n{:6.2f}{:6.2f}{:6.2f}{:6.2f}{:6.2f}".format(100. * ODIN_results["PoT"]["TNR"],
                                                                      100. * ODIN_results["PoT"]["AUROC"],
                                                                      100. * ODIN_results["PoT"]["DTACC"],
                                                                      100. * ODIN_results["PoT"]["AUIN"],
                                                                      100. * ODIN_results["PoT"]["AUOUT"],
                                                                      )
                f.write(result)
                if ODIN_best_tnr[out_count] < ODIN_results['PoT']['TNR']:
                    ODIN_best_results[out_count] = ODIN_results
                    ODIN_best_tnr[out_count] = ODIN_results['PoT']['TNR']
                    ODIN_best_temperature[out_count] = temperature
                    ODIN_best_magnitude[out_count] = magnitude

    f.close()
    
    # print the results
    # print('Baseline method: in_distribution: ')
    # count_out = 0
    # for results in base_line_list:
    #     for mtype in mtypes:
    #         print(' {mtype:6s}'.format(mtype=mtype), end='')
    #     print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
    #     print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
    #     print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
    #     print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
    #     print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
    #     print('')
    #     count_out += 1
        
    print('ODIN method: in_distribution: ')
    count_out = 0
    for results in ODIN_best_results:
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        # print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        # print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        # print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        # print('temperature: ' + str(ODIN_best_temperature[count_out]))
        # print('magnitude: '+ str(ODIN_best_magnitude[count_out]))
        print('')
        count_out += 1
        best_TNR = results['PoT']['TNR']
        best_AUROC = results['PoT']['AUROC']

    return best_TNR, best_AUROC

    
# if __name__ == '__main__':
#     main()