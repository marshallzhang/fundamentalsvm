import xlrd, os, time, string

# setup for getting rid of non digits
all = string.maketrans('','')
nodigs = all.translate(all, string.digits)

book = xlrd.open_workbook("raw_excel/1-50.xls")
sh = book.sheet_by_index(0)

out = open('legend.txt','w')

labels = []
rownums = []
for ratio in [152,153,154,155,156,157,158,159,160,161,162,163,164,167,168,169,170,171,172,173,176,177,178,179,180,181,184,185,186,187,188,189,190]:
    ratio = ratio - 1
    label = sh.cell_value(ratio,0)
    labels.append(label + ' ' + str(ratio))
    rownums.append(ratio)

for metric in [47,60,62,65,66,67,68,69,71,74,75,78,79,83,84,88,91,92,93,104,105,107,109,111,113,117,118,122,128,138,140]:
    metric -= 1
    label = sh.cell_value(metric,0)
    labels.append("d" + label + ' ' + str(metric))
    rownums.append(metric)

labels.append("dGross profit minus dSales")
labels.append("dSales minus dOther OpEx")
labels.append("dSales minus dCurrent Assets")
labels.append("Income increased or decreased")
              
for i in range(1,len(labels)+1):
    out.write(str(i) + " " + labels[i-1] + '\n')

out.close()

