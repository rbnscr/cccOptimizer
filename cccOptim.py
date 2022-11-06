import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk

from itertools import repeat
from scipy import interpolate
from tkinter import filedialog

from utils import *
from optim import *


DATA_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(DATA_DIR, "save")


class CCCOptimizer:
    def __init__(
        self, modules: dict, groups: dict, bounds, eval_range, extrplMode: str
    ) -> None:
        self.allModules = modules
        self.groups = groups
        self.bounds = bounds
        self.eval_range = eval_range
        self.extrplMode = extrplMode
        self._interpolate_curve_input_data()
        self._modules_per_groups()

    def plot_char_curves(self):
        f = plt.figure()
        f.add_subplot(1, 1, 1)
        for key in self.allModules.keys():
            x = self.allModules[key][0]
            y = self.allModules[key][1]
            plt.plot(x, y)
        plt.grid()
        return f

    def _interpolate_curve_input_data(self):
        if self.extrplMode == "none":
            for key in self.allModules.keys():
                x = self.allModules[key][0]
                y = self.allModules[key][1]
                if all([x[0] != 0, y[0] != 0]):
                    x = np.hstack(
                        (
                            self.bounds[0],
                            np.min(x) - 1,
                            x,
                            np.max(x) + 1,
                            self.bounds[1],
                        )
                    )
                    y = np.hstack([0, 0, y, 0, 0])
                    newArray = np.vstack([x, y])
                    self.allModules[key] = newArray
        elif self.extrplMode == "linear":
            for key in self.allModules.keys():
                x = self.allModules[key][0]
                y = self.allModules[key][1]
                if all([x[0] != 0, y[0] != 0]):
                    # Calc zero crossing at start and end
                    zero_cr_start = zero_crossing(x[0:2], y[0:2])
                    zero_cr_end = zero_crossing(x[-2:], y[-2:])
                    # Calc y at bounds
                    y_cr_start = lin_y(self.bounds[0], x[0:2], y[0:2])
                    y_cr_end = lin_y(self.bounds[1], x[-2:], y[-2:])
                    x = np.hstack(
                        (
                            self.bounds[0],
                            np.max([self.bounds[0], zero_cr_start]),
                            x,
                            np.min([self.bounds[1], zero_cr_end]),
                            self.bounds[1],
                        )
                    )
                    y = np.hstack(
                        [0, np.max([0, y_cr_start]), y, np.max([0, y_cr_end]), 0]
                    )
                    newArray = np.vstack([x, y])
                    self.allModules[key] = newArray

    def setup_interpolation_function(self, groupKey):
        numberOfFunctions = self.nModules[groupKey]  # Number of modules per group
        moduleKeys = self.groups[groupKey]
        f = []
        for i in range(numberOfFunctions):
            x = self.allModules[moduleKeys[i]][0]
            y = self.allModules[moduleKeys[i]][1]
            f.append(interpolate.interp1d(x, y, fill_value="extrapolate"))
        return f

    def set_group_curve(self, d: dict):
        self.group_curve = d

    def write_group_curve(self):
        df = pd.DataFrame.from_dict(self.group_curve, orient="index")
        df.to_string(f"{SAVE_DIR}{os.path.sep}groupCurve.txt")

    def set_module_portion(self, d: dict):
        self.module_portion = d

    def write_module_portion(self, fn=f"{SAVE_DIR}{os.path.sep}modulePortion.txt"):
        outputString = "group; module; "
        for elem in self.eval_range:
            outputString = outputString + f"{elem};"
        outputString = outputString + "\n"
        for key in self.module_portion.keys():
            for i in range(len(self.module_portion[key][0])):
                st = f"{key};{self.groups[key][i]};"
                for j in range(len(self.module_portion[key])):
                    st = st + f"{self.module_portion[key][j][i]};"
                outputString = outputString + st + "\n"

        with open(fn, "w") as f:
            f.write(outputString)

    def _modules_per_groups(self):
        lng = {}
        for key in self.groups.keys():
            lng[key] = len(self.groups[key])
        self.nModules = lng


def handle_args():
    parser = argparse.ArgumentParser(description="Bub")
    parser.add_argument(
        "-file",
        dest="filepath",
        type=str,
        required=False,
        help="Data filepath, if None -> file picker",
    )
    parser.add_argument(
        "-plot",
        dest="plotFlag",
        required=False,
        action="store_true",
        help="Save all curves as plot (default: False)",
    )
    parser.add_argument(
        "-plot-results",
        dest="plotResults",
        required=False,
        action="store_true",
        help="Plots resulting curve (default: False)",
    )
    parser.add_argument(
        "-template",
        dest="templateTuple",
        required=False,
        help="Creates template .json file with tuple (#modules,#groups)",
        type=tuple_type,
    )

    return parser.parse_args()


def create_directory_structure():
    """
    Creates the directory structure if it doesn't already exist.
    """
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)


def check_bounds(mods: dict, bounds: list) -> bool:
    check = False
    # True = Out of bounds
    maxBoundary = np.max(bounds)
    for key in mods.keys():
        maxValue = np.max(mods[key])
        if maxValue > maxBoundary:
            check = True
    return check


def write_dict_to_csv(d: dict):
    df = pd.DataFrame.from_dict(d)
    df.to_csv(SAVE_DIR + f"{os.path.sep}outputData.csv")


def write_dict_to_file(d: dict, fn: str):
    outputString = ""
    for key in d.keys():
        for i in range(len(d[key][0])):
            st = f"{key};"
            for j in range(len(d[key])):
                st = st + f"{d[key][j][i]};"
            outputString = outputString + st + "\n"

    with open(fn, "w") as f:
        f.write(outputString)


def create_template_json(templateDims: tuple):

    moduleStr = ""
    for i in range(templateDims[0]):
        if i == max(range(templateDims[0])):
            moduleStr = moduleStr + f'"module{i}":[[<x_values>],[<y_values>]]\n'
        else:
            moduleStr = moduleStr + f'"module{i}":[[<x_values>],[<y_values>]],\n'

    groupStr = ""
    for i in range(templateDims[1]):
        if i == max(range(templateDims[1])):
            groupStr = groupStr + f'"group{i}":[<modules>]\n'
        else:
            groupStr = groupStr + f'"group{i}":[<modules>],\n'

    st = (
        '{\n"modules":{\n'
        + moduleStr
        + '},\n\n"groups":{\n'
        + groupStr
        + '},\n\n"eval_range":[0,...,<max>],\n\n"bounds":[0,<max>],\n\n"extrapolation":<none|linear>\n}'
    )

    with open(DATA_DIR + f"{os.path.sep}template.json", "w") as text_file:
        text_file.write(st)

    print(f"template.json file has been created in {DATA_DIR}")


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def check_extrapolation_mode(mode: str) -> bool:
    if all([mode != "none", mode != "linear"]):
        return True


def Main():
    args = handle_args()

    if args.templateTuple:
        create_template_json(args.templateTuple)
        sys.exit()

    if not args.filepath:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        f = open(file_path)
    else:
        f = open(args.filepath)
    parsedData = json.load(f)

    bounds = parsedData["bounds"]
    modules = parsedData["modules"]
    groups = parsedData["groups"]
    eval_range = parsedData["eval_range"]

    extrplMode = parsedData["extrapolation"]
    extrplMode = extrplMode.lower()
    if check_extrapolation_mode(extrplMode):
        print("Extrapolation mode in .json has to be none, or linear.")
        sys.exit()

    if check_bounds(modules, bounds):
        print("Bounds are not sufficiently set. Exiting program.")
        sys.exit()

    cc = CCCOptimizer(modules, groups, bounds, eval_range, extrplMode)
    cc._modules_per_groups()

    if args.plotFlag:
        fig = cc.plot_char_curves()
        fig.savefig(SAVE_DIR + f"{os.path.sep}all-curves.png")
        sys.exit()

    # Optimization loop starts here
    b = (np.min(bounds), np.max(bounds))
    cons = {"type": "eq", "fun": value_constraint, "args": (5,)}

    x = eval_range
    dictOutput = {}
    dictRelPortionOutput = {}

    for key in groups.keys():
        print(f"Start optimizing group {key}")
        y = []
        relativePortionOfModule = []
        interpolationFunction = cc.setup_interpolation_function(key)
        bnds = tuple(repeat(b, len(interpolationFunction)))
        for eval_value in eval_range:
            cons = {"type": "eq", "fun": value_constraint, "args": (eval_value,)}
            sol = multistart_minimize(
                objective,
                interpolationFunction,
                bnds,
                cons,
                len(interpolationFunction),
                40,
            )
            y.append(-objective(sol.x, interpolationFunction))
            relativePortionOfModule.append(sol.x)
        dictOutput[key] = y
        dictRelPortionOutput[key] = relativePortionOfModule
        plt.plot(x, y)

    # Write to files
    cc.set_group_curve(dictOutput)
    cc.write_group_curve()
    cc.set_module_portion(dictRelPortionOutput)
    cc.write_module_portion()

    if args.plotResults:
        plt.grid()
        plt.show()


if __name__ == "__main__":
    create_directory_structure()
    Main()
