import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace


parser = argparse.ArgumentParser(
    description='DataAnalysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input-file', type=str,
                    default='multiple_tests.txt', help='File with input data', metavar='PATH')
parser.add_argument('--pictures-format', type=str, default='.png',
                    help='Format of output pictures, with a trailing .')


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

            data_dict[network][dataset][rmethod][scaling_factor].add(
                (tuple(wbits_1), tuple(actbits_1), tuple(drbits_1), acc_1, mem_red_1))

            if len(line) > 8:
                wbits_2, actbits_2, drbits_2, acc_2, mem_red_2 = line[8:]
                data_dict[network][dataset][rmethod][scaling_factor].add(
                    (tuple(wbits_2), tuple(actbits_2), tuple(drbits_2), acc_2, mem_red_2))

    f.close()

    return data_dict


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
                    ax.scatter(x, y, color=colors[i],
                               label=str(scaling_factor))

                ax.legend()
                ax.grid(True)
                ax.set_title('_'.join([network, dataset, rmethod]))
                ax.set_xlabel('Weight memory reduction')
                ax.set_ylabel('Accuracy')

                fig.savefig('figures/scaling_factors_analysis/' +
                            '_'.join([network, dataset, rmethod]) + '.png')


def main():
    args = parser.parse_args()
    data_analysis(args)


if __name__ == '__main__':
    main()
