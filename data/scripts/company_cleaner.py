import os

for file in os.listdir('../company_data/'):
    f = open('../company_data/' + file, 'r')
    print file
    lines = [line for line in f.readlines()]
    print lines
    try:
        if float(len([line for line in lines if line != '\n']))/float(len(lines)) > 0.95 and lines[-1] != '\n':
            o = open('../good_company_data/' + file, 'w')
            for line in lines:
                o.write(line)
            o.close()
    except:
        pass
    f.close()
