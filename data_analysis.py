import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import linspace
from typing import List, Tuple


parser = argparse.ArgumentParser(
    description='DataAnalysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input-file', type=str,
                    default='multiple_tests.txt', help='File with input data', metavar='PATH')
parser.add_argument('--pictures-format', type=str, default='png',
                    help='Format of output pictures, with a trailing .')
parser.add_argument('--scaling-factor-analysis',
                    action='store_true', default=False)
parser.add_argument('--quantization-method-analysis',
                    action='store_true', default=False)
parser.add_argument('--output-file', type=str, default='./final_results.txt')


ACT_SIZES = {}
ACT_SIZES['ShallowCapsNet'] = {
    'mnist': [784, 102400, 9216, 160], 'fashion-mnist': [784, 102400, 9216, 160]}
ACT_SIZES['DeepCaps'] = {'mnist': [784, 100352, 100352, 50176, 16384, 4096, 320],
                         'fashion-mnist': [784, 100352, 100352, 50176, 16384, 4096, 320],
                         'cifar10': [12288, 524288, 524288, 262144, 65536, 16384, 320]}

DATASET_TITLES = {'mnist': 'MNIST',
                  'fashion-mnist': 'FMNIST', 'cifar10': 'CIFAR10'}

SF = 'MAX'  # 'MAX', 'OPT'

scaling_factors_dict = {}
if SF == 'OPT':
    scaling_factors_dict["ShallowCapsNet"] = {
        'mnist': 6.0, 'fashion-mnist': 8.0}
    scaling_factors_dict["DeepCaps"] = {
        'mnist': 6.0, 'fashion-mnist': 8.0, 'cifar10': 8.0}
elif SF == 'MAX':
    scaling_factors_dict["ShallowCapsNet"] = {
        'mnist': 10000.0, 'fashion-mnist': 10000.0}
    scaling_factors_dict["DeepCaps"] = {
        'mnist': 10000.0, 'fashion-mnist': 10000.0, 'cifar10': 10000.0}


def analyze_file(filename):
    f = open(filename, "r")

    # dict struct
    # {network:
    #          {dataset:
    #                   quant_method:
    #                               scaling_factor: [mem_red, acc, [wbits], [act_bits], [dr_bits]]}}

    data_dict = {}

    while True:
        line = f.readline()
        if not line:
            break

        if "ShallowCapsNet" in line or "DeepCaps" in line:
            network, dataset, rmethod = line.replace(
                '\t', ' ').replace('\n', '').split(' ')
            #data_dict[network] = {dataset: {rmethod: {}}}
            if network not in data_dict.keys():
                data_dict[network] = {}
            if dataset not in data_dict[network].keys():
                data_dict[network][dataset] = {}
            if rmethod not in data_dict[network][dataset].keys():
                data_dict[network][dataset][rmethod] = {}
            #print(network, dataset, rmethod)
        else:
            line = line.replace('\t', ' ').replace(
                '(', '').replace(')', '').replace('\n', '')
            remove_comma = True
            remove_space = False
            len_line = len(line)
            i = 0
            while i < len_line:
                if line[i] == '[':
                    remove_comma = False
                    remove_space = True
                    i += 1
                elif line[i] == ']':
                    remove_comma = True
                    remove_space = False
                    i += 1
                else:
                    if (line[i] == ',' and remove_comma) or (line[i] == ' ' and remove_space):
                        line = line[0:i] + line[i+1:]
                        len_line -= 1
                    else:
                        i += 1

            line = line.split(' ')
            for j in range(len(line)):
                if j in [0, 1, 2, 6, 7, 11, 12]:
                    line[j] = float(line[j])
                else:
                    line[j] = line[j].replace(
                        '[', '').replace(']', '').split(',')
                    for z in range(len(line[j])):
                        line[j][z] = int(line[j][z])

            scaling_factor, _, _, wbits_1, actbits_1, drbits_1, acc_1, mem_red_1 = line[0:8]

            if scaling_factor not in data_dict[network][dataset][rmethod].keys():
                data_dict[network][dataset][rmethod][scaling_factor] = set()

            if len(line) > 8:
                data_dict[network][dataset][rmethod][scaling_factor].add(
                    (tuple(wbits_1), tuple(actbits_1), tuple(drbits_1), acc_1, mem_red_1, False))
                wbits_2, actbits_2, drbits_2, acc_2, mem_red_2 = line[8:]
                data_dict[network][dataset][rmethod][scaling_factor].add(
                    (tuple(wbits_2), tuple(actbits_2), tuple(drbits_2), acc_2, mem_red_2, False))
            else:
                data_dict[network][dataset][rmethod][scaling_factor].add(
                    (tuple(wbits_1), tuple(actbits_1), tuple(drbits_1), acc_1, mem_red_1, True))

    f.close()

    return data_dict


def pareto_frontier(data_x: List[float], data_y: List[float], maxX: bool = True, maxY: bool = True) -> Tuple[List[float], List[float], List[int]]:
    # Add explicit index of xy pair in the array
    my_list_idx = [[data_x[i], data_y[i], i]
                   for i in range(len(data_x))]
    # Sort the list in either ascending or descending order of X
    my_list_idx = sorted(my_list_idx, reverse=maxX)

    my_list = [[my_list_idx[i][0], my_list_idx[i][1]]
               for i in range(len(my_list_idx))]
    my_idx = [my_list_idx[i][2] for i in range(len(my_list_idx))]

    # Start the Pareto frontier with the first value in the sorted list
    p_front = [my_list[0]]
    p_idx = [my_idx[0]]
    # Loop through the sorted list
    for pair, index in zip(my_list[1:], my_idx[1:]):
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
                p_idx.append(index)
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
                p_idx.append(index)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY, p_idx


def pareto_frontier_3d(costs: List[List[float]]) -> List[bool]:
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = np.asarray(costs)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
            np.any(costs[i+1:] > c, axis=1))
    return is_efficient.tolist()


def data_analysis(args):

    data_dict = analyze_file(args.input_file)

    for key_1 in data_dict.keys():
        print(key_1)
        for key_2 in data_dict[key_1].keys():
            print('\t', key_2)
            for key_3 in data_dict[key_1][key_2].keys():
                print('\t\t', key_3)
                for key_4 in data_dict[key_1][key_2][key_3].keys():
                    print('\t\t\t', key_4, len(
                        data_dict[key_1][key_2][key_3][key_4]))

    ##### SCALING FACTOR ANALYSIS ######
    if args.scaling_factor_analysis:
        for network in data_dict.keys():
            for dataset in data_dict[network].keys():
                for rmethod in data_dict[network][dataset].keys():
                    fig, ax = plt.subplots()
                    number_of_scaling_factors = len(
                        data_dict[network][dataset][rmethod].keys())
                    cm_subsection = linspace(0., 1., number_of_scaling_factors)
                    colors = [cm.jet(x) for x in cm_subsection]
                    for i, scaling_factor in enumerate(data_dict[network][dataset][rmethod].keys()):
                        x, y = [], []
                        for test in data_dict[network][dataset][rmethod][scaling_factor]:
                            x.append(test[4])
                            y.append(test[3])
                        p_frontX, p_frontY, _ = pareto_frontier(
                            x, y, maxX=True, maxY=True)
                        ax.scatter(x, y, color=colors[i])
                        ax.plot(p_frontX, p_frontY, color=colors[i],
                                label=str(scaling_factor))

                    ax.legend()
                    ax.grid(True)
                    ax.set_title('_'.join([network, dataset, rmethod]))
                    ax.set_xlabel('Weight memory reduction')
                    ax.set_ylabel('Accuracy')

                    fig.savefig('figures/scaling_factors_analysis/' +
                                '_'.join([network, dataset, rmethod]) + '.' + args.pictures_format)

    ##### QUANTIZATION METHODS ANALYSIS #####
    if args.quantization_method_analysis:
        for network in data_dict.keys():
            for dataset in data_dict[network].keys():
                fig, ax = plt.subplots()
                number_of_qmethods = len(
                    data_dict[network][dataset].keys())
                cm_subsection = linspace(0., 1., number_of_qmethods)
                colors = [cm.jet(x) for x in cm_subsection]
                # for i, rmethod in enumerate(data_dict[network][dataset].keys()):
                for i, rmethod in enumerate(["round_to_nearest", "stochastic_rounding", "truncation"]):
                    curr_scaling_factor = scaling_factors_dict[network][dataset]
                    x, y = [], []
                    for test in data_dict[network][dataset][rmethod][curr_scaling_factor]:
                        x.append(test[4])
                        y.append(test[3])
                    p_frontX, p_frontY, _ = pareto_frontier(
                        x, y, maxX=True, maxY=True)
                    ax.scatter(x, y)  # , color=colors[i])
                    # , color=colors[i])
                    ax.plot(p_frontX, p_frontY, label=rmethod)

                ax.legend()
                ax.grid(True)
                ax.set_title(f'{network} - {DATASET_TITLES[dataset]}')
                ax.set_xlabel('Weight memory reduction')
                ax.set_ylabel('Accuracy')

                fig.savefig('figures/quantization_method_analysis/' +
                            '_'.join([network, dataset]) + '.' + args.pictures_format)

    ##### OVERALL BEST RESULTS PRINT (excel) #####
    # merge di tutti i rounding methods insieme - plot
    # pareto
    f = open(args.output_file, 'w+')
    f2 = open(args.output_file[:-4]+'_rtne'+args.output_file[-4:], "w+")
    f3 = open(args.output_file[:-4] +
              '_rtne_satisfied'+args.output_file[-4:], "w+")

    for network in data_dict.keys():
        for dataset in data_dict[network].keys():
            fig, ax = plt.subplots()
            curr_scaling_factor = scaling_factors_dict[network][dataset]
            x, y, tests, test_rmethod, satisfied = [], [], [], [], []
            for i, rmethod in enumerate(data_dict[network][dataset].keys()):
                for test in data_dict[network][dataset][rmethod][curr_scaling_factor]:
                    x.append(test[4])
                    y.append(test[3])
                    tests.append(test)
                    test_rmethod.append(rmethod)
                    satisfied.append(test[5])
            p_frontX, p_frontY, p_index = pareto_frontier(
                x, y, maxX=True, maxY=True)
            ax.scatter(x, y, color=cm.jet(0))
            ax.plot(p_frontX, p_frontY, color=cm.jet(1))

            ax.grid(True)
            ax.set_title('_'.join([network, dataset]))
            ax.set_xlabel('Weight memory reduction')
            ax.set_ylabel('Accuracy')

            fig.savefig('figures/' +
                        '_'.join([network, dataset]) + '.' + args.pictures_format)

            f.write(f'{network}, {dataset}\n')
            f2.write(f'{network}, {dataset}\n')
            f3.write(f'{network}, {dataset}\n')
            max_acc = 0
            for test in tests:
                max_acc = max(max_acc, test[3])
            min_acc = max_acc - max_acc*0.02
            # per ogni punto pareto con acc >= acc_max meno 2%:
            # print acc, mem red, act volume reduction, distr pesi, attivationi, dr bits
            act_red_list = []
            for test in tests:
                act_32 = sum(ACT_SIZES[network][dataset]) * 32
                act_curr = [
                    a*b for a, b in zip(ACT_SIZES[network][dataset][1:], test[1])]
                act_curr = sum(act_curr) + \
                    ACT_SIZES[network][dataset][0]*test[1][0]
                act_red = act_32 / act_curr
                act_red_list.append(act_red)
            # test: w_bits, a_bits, dr_bits, acc, w_red, a_red
            costs = [[t[3], t[4], a] for t, a in zip(tests, act_red_list)]
            are_pareto = pareto_frontier_3d(costs)

            acc_list_plot = []
            w_red_list_plot = []
            act_red_list_plot = []

            for test, a, p, rmethod, s in zip(tests, act_red_list, are_pareto, test_rmethod, satisfied):
                # if p and test[3] >= min_acc:
                if True:
                    f.write(
                        f'{test[3]:.2f}\t{test[4]:.2f}\t{a:.2f}\t{test[0]}\t{test[1]}\t{test[2]}\n')

                    if rmethod == 'round_to_nearest':
                        f2.write(
                            f'{test[3]:.2f}\t{test[4]:.2f}\t{a:.2f}\t{test[0]}\t{test[1]}\t{test[2]}\n')
                        if s:
                            f3.write(
                                f'{test[3]:.2f}\t{test[4]:.2f}\t{a:.2f}\t{test[0]}\t{test[1]}\t{test[2]}\n')
                        acc_list_plot.append(test[3])
                        w_red_list_plot.append(test[4])
                        act_red_list_plot.append(a)

            fig, ax = plt.subplots(1, 2, figsize=(6.4, 1.8), sharey=True)

            zipped = zip(w_red_list_plot, act_red_list_plot, acc_list_plot)
            zipped = sorted(zipped)
            w_red_list_plot, act_red_list_plot, acc_list_plot = zip(*zipped)

            cm_subsection = linspace(0., 1., len(acc_list_plot))
            colors = [cm.jet(x) for x in cm_subsection]

            ax[0].scatter(w_red_list_plot, acc_list_plot,
                          marker='o', color=colors)
            ax[1].scatter(act_red_list_plot, acc_list_plot,
                          marker='o', color=colors)

            ax[0].set_xlabel("W mem reduction")
            ax[0].set_ylabel("Accuracy (%)")
            ax[0].grid(True)

            ax[1].set_xlabel("A vol reduction")
            ax[1].grid(True)

            fig.suptitle(
                f'{network.replace("Net", "")} - {DATASET_TITLES[dataset]}')

            fig.savefig('figures/' +
                        '_'.join([network, dataset]) + '_curve.' + args.pictures_format, bbox_inches='tight')

    f.close()
    f2.close()
    f3.close()


def main():
    args = parser.parse_args()
    data_analysis(args)


if __name__ == '__main__':
    main()
