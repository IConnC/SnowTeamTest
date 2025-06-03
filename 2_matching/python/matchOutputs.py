from utils import getList

# Path to output of matchingCode
path1 = './pairedFlakeKey24.txt'
path2 = './pairedFlakeKey25.txt'

allThreeCams = getList(path1, path2, v=1)
print(allThreeCams)

f = open("pairedFlakeKey.txt", 'w')
f.write(allThreeCams)
f.close()
