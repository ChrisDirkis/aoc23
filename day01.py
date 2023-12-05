from itertools import *
from functools import *
from math import * 
import re
from collections import defaultdict 

from grid import *

numwords = ["________", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
numregexes = [re.compile(word) for word in numwords]
digit_re = re.compile(r"(\d)")

aod_day = __file__.split("/")[-1][-5:-3]
data_file_name = "inputs/" + aod_day
testing_file_name = data_file_name + "t"

def part_1(filename):
    print(f"Part 1: {filename}")
    with open(filename) as file:
        lines = [[int(d) for d in digit_re.findall(line)] for line in file]
        values = [line[0] * 10 + line[-1] for line in lines]
        print(sum(values))
        pass

def part_2(filename):
    print(f"Part 2: {filename}")
    with open(filename) as file:
        s = 0
        for line in file:
            line = line.strip()
            orig = line
            for j in range(len(line)):
                for i, word in enumerate(numwords):
                    if line[j:].startswith(word):
                        line = line[:j] + str(i) + line[j + len(word):]
                j = len(line) - j - 1
                for i, word in enumerate(numwords):
                    if line[j:].startswith(word):
                        line = line[:j] + str(i) + line[j + len(word):]

            print(orig, line)
            digits = [int(d) for d in digit_re.findall(line)] 
            value = 10 * digits[0] + digits[-1]
            s += value
        print(s)
        pass

if __name__ == "__main__":
    #part_1(testing_file_name)
    #part_1(data_file_name)
    part_2(testing_file_name)
    part_2(data_file_name)