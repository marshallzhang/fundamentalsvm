import os

for file in os.listdir('../good_company_data/'):
    print file
    f = open('../good_company_data/' + file, 'r')
    lines = [line for line in f.readlines()]
    if len(lines) != 53:
        continue
    o = open('../final_company_data/' + file, 'w')
    for line in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
        o.write(lines[line-1])
    o.close()
    f.close()
