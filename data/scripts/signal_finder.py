import os

signals = []

def readlineiterator(file, line, signal):
    if line > 0:
        signal = file.readline()
        return readlineiterator(file, line - 1, signal)
    else:
        return signal
        
def file_iterator(line):
#    signal = ''
#    signal = readlineiterator(open('../company_data/URSTADT-BIDDLE-PROPERTIES-INC_31-Oct-2008.txt'), line, signal)
#    print repr(signal)
#    if signal:
#        return True
#    return False 
    for file in os.listdir('../good_company_data'):
        f = open('../good_company_data/' + file)
        signal = ''
        signal = readlineiterator(f, line, signal)
        if signal == '\n':
            f.close()
            return False
        f.close()
    return True

for line in range(1,53):
    if file_iterator(line):
        signals.append(line)

print signals
