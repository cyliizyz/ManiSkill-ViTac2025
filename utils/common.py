import copy
import os
import sys
import time
from datetime import datetime

import numpy as np
import cv2


def generate_patch_array(super_resolution_ratio=10):
    circle_radius = 3
    size_slot_num = 50
    base_circle_radius = 1.5

    patch_array = np.zeros(
        (super_resolution_ratio, super_resolution_ratio, size_slot_num, 4 * circle_radius, 4 * circle_radius),
        dtype=np.uint8)
    for u in range(super_resolution_ratio):
        for v in range(super_resolution_ratio):
            for w in range(size_slot_num):
                img_highres = np.ones(
                    (4 * circle_radius * super_resolution_ratio, 4 * circle_radius * super_resolution_ratio),
                    dtype=np.uint8) * 255
                center = np.array(
                    [circle_radius * super_resolution_ratio * 2, circle_radius * super_resolution_ratio * 2],
                    dtype=np.uint8)
                center_offseted = center + np.array([u, v])
                radius = round(base_circle_radius * super_resolution_ratio + w)
                img_highres = cv2.circle(img_highres, tuple(center_offseted), radius, (0, 0, 0), thickness=cv2.FILLED,
                                         lineType=cv2.LINE_AA)
                img_highres = cv2.GaussianBlur(img_highres, (17, 17), 15)
                img_lowres = cv2.resize(img_highres, (4 * circle_radius, 4 * circle_radius),
                                        interpolation=cv2.INTER_CUBIC)
                patch_array[u, v, w, ...] = img_lowres

    return {
        "base_circle_radius": base_circle_radius,
        "circle_radius": circle_radius,
        "size_slot_num": size_slot_num,
        "patch_array": patch_array,
        "super_resolution_ratio": super_resolution_ratio,
    }


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


def get_ms():
    milliseconds = str(int(time.time() * 1000) % 1000)
    if len(milliseconds) == 1:
        return "00" + milliseconds
    if len(milliseconds) == 2:
        return "0" + milliseconds
    return milliseconds


def get_time():
    now = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    milliseconds = get_ms()
    return now + "." + milliseconds


def dump_args(f, arg):
    arg_dict = arg.__dict__
    time_str = get_time()
    f.write("--------------------Arguments--------------------\n")
    f.write("Begin at time : " + time_str + "\n")
    for key, value in arg_dict.items():
        f.write("{:>40}: {:<100}\n".format(key, value))
    f.write("-----------------------End-----------------------\n")


def dump_args_to_tensorboard(tensorboard_writer, arg, global_step=0):
    arg_dict = arg.__dict__
    time_str = get_time()
    string = f"Begin at time : {time_str} \n"

    for key, value in arg_dict.items():
        try:
            str_append = "{:>50}: {:<100}\n".format(key, value)
            string += str_append
        except:
            pass

    tensorboard_writer.add_text(tag="ARGUMENTS", text_string=string, global_step=global_step)


def dump_dict_to_tensorboard(tensorboard_writer, target_dict, global_step=0):
    time_str = get_time()
    string = f"Log time : {time_str} \n"

    def append_dict_to_string(target_string, target_dict, prefix=""):
        for key, value in target_dict.items():
            if type(value) is dict:
                target_string = append_dict_to_string(target_string, value, prefix=f"{key}.")
            else:
                try:
                    str_append = "{:>50}: {:<100}\n".format(prefix + key, value)
                    target_string += str_append
                except:
                    pass
        return target_string

    string = append_dict_to_string(string, target_dict)

    tensorboard_writer.add_text(tag="ARGUMENTS", text_string=string, global_step=global_step)


def copy_args(source, target):
    source_dict = source.__dict__
    target_dict = target.__dict__
    for key, value in source_dict.items():
        if key in target_dict.keys():
            target.__setattr__(key, value)


class Params:
    def __init__(self):
        pass

    def __str__(self):
        _dict = self.__dict__
        content = "-----------Parameters-----------------\n"
        for key in sorted(_dict.keys()):
            value = _dict[key]
            try:
                content += "{:>30}: {:<60}\n".format(key, value)
            except Exception as e:
                continue
        content += "-----------------------End-----------------------\n"
        return content

    def parse_from_file(self, file_name):
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                split = line.split(":")
                if len(split) == 2:
                    param_name = split[0].replace(" ", "").replace("\n", "")
                    value = split[1].replace(" ", "").replace("\n", "")
                    if hasattr(self, param_name):
                        if isinstance(getattr(self, param_name), str):
                            setattr(self, param_name, value)
                        else:
                            setattr(self, param_name, float(value))


def randomize_params(lower_bound: Params, upper_bound: Params):
    random_param = copy.deepcopy(lower_bound)
    for vvv in random_param.__dict__.keys():
        lb = lower_bound.__getattribute__(vvv)
        ub = upper_bound.__getattribute__(vvv)
        if type(lb) is str:
            if lb != ub:
                raise Exception("Strings do not match")
            random_param.__setattr__(vvv, lb)
        elif type(lb) is tuple:
            random_param.__setattr__(vvv, lb)
        elif type(lb) is list:
            random_param.__setattr__(vvv, lb)
        else:
            random_value = np.random.rand(1)[0] * (ub - lb) + lb
            random_param.__setattr__(vvv, random_value)
    return random_param


def get_average_params(lower_bound: Params, upper_bound: Params):
    average_param = copy.deepcopy(lower_bound)
    for vvv in average_param.__dict__.keys():
        lb = lower_bound.__getattribute__(vvv)
        ub = upper_bound.__getattribute__(vvv)
        if type(lb) is str:
            if lb != ub:
                raise Exception("Strings do not match")
            average_param.__setattr__(vvv, lb)
        elif type(lb) is tuple:
            average_param.__setattr__(vvv, lb)
        elif type(lb) is list:
            average_param.__setattr__(vvv, lb)
        else:
            average_value = 1 / 2.0 * (ub + lb)
            average_param.__setattr__(vvv, average_value)
    return average_param


def check_whether_samples_generated(folder: str, ids: list, prefix="", suffix=".npy"):
    generated = True
    not_generated_list = []
    for id in ids:
        if not os.path.exists(os.path.join(folder, prefix + str(id) + f"{suffix}")):
            generated = False
            not_generated_list.append(id)
    return generated, not_generated_list
