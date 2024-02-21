
from collections import OrderedDict
import pandas as pd
import numpy as np


class Dtype:
    fn = None
    
    def __init__(self, name) -> None:
        self.name = name
    
    def __call__(self, value):
        if value == "?":
            return pd.NA
        return self.__class__.fn(value)


class Integer(Dtype):
    fn = lambda x: int(x)


class String(Dtype):
    fn = lambda x: x


class Real(Dtype):
    fn = lambda x: float(x)



class OpenMLDataset:

    def __init__(self, fname) -> None:
        self.fname = fname
        self.attrs = OrderedDict()

    def load(self):
        with open(self.fname) as f:
            data = f.readlines()
        # Load Attributes
        for i, line in enumerate(data):
            if "@ATTRIBUTE" in line:
                _, att_name, att_type = line.strip().split(" ")
                self.set_att_type(att_name, att_type)
            if "@DATA" in line:
                break
        i += 1
        transformed_data = []
        for j in range(i, len(data)):
            line = data[j]
            transformed_data.append(self.transform(line))

        return pd.DataFrame(transformed_data, columns=list(self.attrs.keys()))
    
        
    def transform(self, line):
        items = []
        _line = line
        for _, dtype in self.attrs.items():
            if line.startswith("'"):
                idx = 0
                while True:
                    match_idx = line.find("'", idx + 1, len(line))
                    idx = match_idx
                    if line[match_idx - 1] == "\\":
                        continue
                    else:
                        break
                idx = idx + 1
            else:
                idx = line.find(",")
                if idx == -1:
                    idx = len(line)
            
            item = line[0:idx].strip()
            if item[0] == "'":
                item = item[1:-1]
            try:
                items.append(dtype(item))
            except Exception as e:
                print(_line)
                raise e
            line = line[idx + 1:]
        return items 

    def set_att_type(self, name, atype):
        if name in self.attrs:
            raise Exception("Key already present")
        if atype.lower() == "integer":
            self.attrs[name] = Integer(name)
        elif atype.lower() == "string":
            self.attrs[name] = String(name)
        elif atype.lower() == "real":
            self.attrs[name] = Real(name)
        else:
            raise NotImplementedError
