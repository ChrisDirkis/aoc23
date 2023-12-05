from itertools import *
from functools import *
from math import * 
import re
from collections import defaultdict 

from grid import *

digit_re = re.compile(r"(-?\d+)")

aod_day = __file__.split("/")[-1][-5:-3]
data_file_name = "inputs/" + aod_day
testing_file_name = data_file_name + "t"

def part_1(filename):
    print(f"Part 1: {filename}")

    rm, gm, bm = 12, 13, 14
    with open(filename) as file:
        s = 0
        for i,game in enumerate(file):
            game = game.split(": ")[1]
            segments = game.split("; ")
            possible = True
            for segment in segments:
                rma = re.findall(r"(\d+) red", segment)
                red = int(rma[0]) if len(rma) > 0 else 0 

                gma = re.findall(r"(\d+) green", segment)
                green = int(gma[0]) if len(gma) > 0 else 0

                bma = re.findall(r"(\d+) blue", segment)
                blue = int(bma[0]) if len(bma) > 0 else 0

                if red > rm or blue > bm or green > gm:
                    possible = False
                    break
            if possible:
                s += i + 1
        print(s)
        pass

def part_2(filename):
    print(f"Part 2: {filename}")
    with open(filename) as file:        
        s = 0
        for i,game in enumerate(file):
            game = game.split(": ")[1]
            segments = game.split("; ")
            r, g, b = 0, 0, 0
            for segment in segments:
                rma = re.findall(r"(\d+) red", segment)
                red = int(rma[0]) if len(rma) > 0 else 0 

                gma = re.findall(r"(\d+) green", segment)
                green = int(gma[0]) if len(gma) > 0 else 0

                bma = re.findall(r"(\d+) blue", segment)
                blue = int(bma[0]) if len(bma) > 0 else 0

                r = max(red, r)
                g = max(green, g)
                b = max(blue, b)

            power = r*g*b
            s += power
        print(s)

if __name__ == "__main__":
    part_1(testing_file_name)
    part_1(data_file_name)
    part_2(testing_file_name)
    part_2(data_file_name)