from pathlib import Path
import argparse
import numpy as np
import re


def add_section(sec_name, sec_dict):
    if sec_dict.get(sec_name) is None:
        sec_dict.update({sec_name:[]})

    return sec_dict


def add_opt(opt, sec_dict):
    sec = opt.split(".")[0]
    val = opt.split(".")[1]

    if sec_dict.get(sec) is None:
        sec_dict.update({sec:[val]})
    else:
        sec_dict.get(sec).append(val)

    return sec_dict


def remove_opt(opt, sec_dict):
    sec = opt.split(".")[0]
    val = opt.split(".")[1]
    
    if sec_dict.get(sec) is None:
        raise RuntimeError("Section " + " sec not existing.")
    else:
        temp_list = sec_dict.get(sec)
        for i in range(len(temp_list)):
            if len(re.findall(val + r"[=].+", temp_list[i])) != 0:
                sec_dict.get(sec).remove(temp_list[i])
                break

    return sec_dict


def remove_section(sec_name, sec_dict):
    sec_dict.pop(sec_name, None)

    return sec_dict


def read_operations_from_file(filename):
    op_list = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        op_list.append([line.split()[0].strip(), line.split()[1].strip()])

    return op_list


def is_valid_op(op):
    if (op[0] == 'rs'):
        return True
    elif (op[0] == 'ro'):
        if (len(op[1].split('.')) == 2):
            return True
    elif (op[0] == 'as'):
        return True
    elif (op[0] == 'ao'):
        regex = ".+[.].+[=].+"
        if (len(re.findall(regex, op[1])) == 0):
            print()
            return False
    else:
        return False

    return True


def output_sec_dict(sec_dict, orig_config):
    f = open(orig_config, "w")
    for sec in sec_dict.keys():
        f.write("[" + sec + "]" + "\n")
        for opt in sec_dict.get(sec):
            f.write(opt + "\n")
        f.write("\n")
    f.close()


def set_args():
    parser = argparse.ArgumentParser(
        description='Modify existing configs',
        prog='config_modifier',
        formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('orig_config', metavar='orig_config',
                        type=str, help='Original configuration file location')
    mode.add_argument('--operations', '-o', action='extend', nargs='+',
                        type=str,
                        help='Operations to be conducted:\n' +
                             'remove section: rs <section>\n' +
                             'remove option: ro <section>.<option>\n' +
                             'add section: as <section>\n' +
                             'add option: ao <section>.<option>' +
                             '=<default_value>')
    mode.add_argument('--file', '-f', metavar='file', type=str,
                        help='Read operations from designated file')

    return parser


def parse_args(args):
    if not Path(args.orig_config).is_file():
        raise RuntimeError(str(args.orig_config) + "not found!")
    arg_dict = vars(args)
    if args.file is None:
        op_list = np.array(arg_dict.get("operations"))
        op_list = np.reshape(op_list, (-1,2))
        for op in op_list:
            if not is_valid_op(op):
                raise RuntimeError("Invalid operation: " + " ".join(op))
    else:
        op_list = read_operations_from_file(args.file)

    return arg_dict, op_list


def parse_actions(orig_config, op_list):
    f = open(orig_config, "r")
    lines = f.readlines()
    sec_dict = {}
    curr_sec = None

    for line in lines:
        if (len(re.findall("[[].+[]]", line)) != 0):
            curr_sec = line[1:-2]
            continue
    
        if line == '\n':
            continue

        if sec_dict.get(curr_sec) is None:
            sec_dict.update({curr_sec:[line.strip()]})
        else:
            sec_dict.get(curr_sec).append(line.strip())
    f.close()

    for op in op_list:
        if op[0] == "rs":
            sec_dict = remove_section(op[1], sec_dict)
        elif op[0] == "ro":
            sec_dict = remove_opt(op[1], sec_dict)
        elif op[0] == "as":
            sec_dict = add_section(op[1], sec_dict)
        elif op[0] == "ao":
            sec_dict = add_opt(op[1], sec_dict)
        else:
            raise RuntimeError("Invalid opt: " + " ".join(op))

    return sec_dict

def main():
    parser = set_args()
    #parser.parse_args('./dcpl -o rs a ro b ao c'.split())
    args = parser.parse_args()
    arg_dict, op_list = parse_args(args)
    sec_dict = parse_actions(args.orig_config, op_list)
    output_sec_dict(sec_dict, args.orig_config)



if __name__ == "__main__":
    main()

#"-o", "ao", "loss.loss=SoftMax", "ro", "training.loss", 