import json
from os import listdir
from os.path import isfile
from os.path import join

def ReadGroundTruth(FileName):
    with open(FileName) as f:
        data = f.read()
    
    data = data.replace('\'', '\"')

    return json.loads(data)

def ReadFrequencyData(FileName):
    res = []

    with open(FileName) as f:
        data = f.readlines()

    count = 0
    for line in data:
        line = line.replace('\'', '\"')

        if count % 2:
            res[count // 2].update(json.loads(line))
        else:
            res.append(json.loads(line))
        
        count += 1

    return res

def ReadTextData(Path):
    fileNames = [f for f in listdir(Path) if isfile(join(Path, f))]

    res = dict()

    for fileName in fileNames:
        with open(Path + fileName) as f:
            text = f.read()
            res[fileName] = text

    return res
            
def Layer1(x, y, classifier):
    pass

def Layer2(x, y, classifier):
    pass

def Layer3(x, y, classifier):
    pass

def main():
    ground_truth = ReadGroundTruth('./ground_truth.txt')
    data_frequency = ReadFrequencyData('./out_data.txt')
    data_training = ReadTextData('./Traning Data/')
    data_testing = ReadTextData('./Testing Data/')

    print(data_frequency)

    print('finished')


if __name__ == "__main__":
    main()