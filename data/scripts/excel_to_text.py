import xlrd, os, time, string

# setup for getting rid of non digits
all = string.maketrans('','')
nodigs = all.translate(all, string.digits)

for file in os.listdir('../raw_excel'):
    if 'Store' in file: continue
    book = xlrd.open_workbook('../raw_excel/' + file)
    
    # for sheets
    for sheet in book.sheets():
        company_name = sheet.cell_value(0,1)
        company_name = company_name.replace(' ','-').replace(',','').replace('/','').replace('.','')
       
        try:
            if 'ROE using P/L before tax' not in sheet.cell_value(151,0) or  'Number of employees' not in sheet.cell_value(46,0) or 'P/L for period' not in sheet.cell_value(127,0):
                continue
        except IndexError:
            continue
        
        date_cols = []

        # get dates
        for col in range(0,sheet.ncols):
            date = sheet.cell_value(26,col)

            # for each year
            if '20' in date:
                date_cols.append(col)
        
        year_col_index = -1
        
        for year_col in date_cols:
            year_col_index += 1
            date = sheet.cell_value(26,year_col)
            date = time.strptime(date,"%d %b %Y")
            date = time.strftime("%d-%b-%Y",date)
            

            # get ratios
            if year_col_index < len(date_cols) - 2:
                
                print "DATE"
                print date
                print company_name
                print file
                out = open('../company_data/' + company_name + '_' + date + '.txt', 'w')

                previous_year_col = date_cols[year_col_index + 1]
                data_col = previous_year_col - 1
                ratio_string = ''
                for ratio in [152,153,154,155,156,157,158,159,160,161,162,163,164,172,173,176,178,179,180]:
                    ratio = ratio - 1
                    value = sheet.cell_value(ratio,data_col)
                    try:
                        test = value / value
                        try:
                            ratio_string += str(value) + '\n'
                        except ZeroDivisionError:
                            ratio_string += '\n'
                    except:
                        ratio_string += '\n'
                print ratio_string

                # get delta indicators
                p_previous_year_col = date_cols[year_col_index + 2]
                p_previous_year_data_col = p_previous_year_col - 1
                
                # year over year indicators
                delta_string = ''
                for metric in [60,62,65,66,67,68,69,71,74,75,78,79,83,84,88,91,92,93,104,105,107,109,111,113,117,118,122,128,138,140]:
                    metric = metric - 1
                    previous_year_metric = sheet.cell_value(metric,data_col)
                    p_previous_year_metric = sheet.cell_value(metric,p_previous_year_data_col)
                    try:
                        test = previous_year_metric / p_previous_year_metric
                        try:
                            delta = ((float(previous_year_metric) - float(p_previous_year_metric)) / float(p_previous_year_metric))
                            delta_string += str(delta) + '\n'
                        except ZeroDivisionError:
                            delta_string += '\n'
                    except:
                        delta_string += '\n'
                print delta_string

                # get double delta indicators
                d_delta_string = ''
                for metric1,metric2 in [(109,105),(105,111),(105,65)]:
                    metric1 = metric1 - 1
                    metric2 = metric2 - 1
                    previous_year_m1 = sheet.cell_value(metric1,data_col)
                    previous_year_m2 = sheet.cell_value(metric2,data_col)
                    p_previous_year_m1 = sheet.cell_value(metric1,p_previous_year_data_col)
                    p_previous_year_m2 = sheet.cell_value(metric2,p_previous_year_data_col)
                    
                    try:
                        test = (previous_year_m1 / previous_year_m2) / (p_previous_year_m1 / p_previous_year_m2)
                        try:
                            delta_m1 = ((float(previous_year_m1) - float(p_previous_year_m1)) / float(p_previous_year_m1))
                            delta_m2 = ((float(previous_year_m2) - float(p_previous_year_m2)) / float(p_previous_year_m2))
                            ddelta = delta_m1 - delta_m2
                            d_delta_string += str(ddelta) + '\n' 
                        except ZeroDivisionError:
                            d_delta_string += '\n'
                    except:
                        d_delta_string += '\n'
                print d_delta_string
                
                # get earnings increase
                current_year_earnings = sheet.cell_value(127,year_col - 1)
                previous_year_earnings = sheet.cell_value(127,data_col)
                try:
                    test = current_year_earnings / previous_year_earnings
                    try:
                        if current_year_earnings - previous_year_earnings >= 0:
                            earnings_increase = 1
                        else:
                            earnings_increase = -1
                    except ZeroDivisionError:
                        earnings_increase = ''
                except:
                    earnings_increase = ''

                # put it all together and write
                bigstring = ratio_string + delta_string + d_delta_string + str(earnings_increase)
                out.write(bigstring)
                out.close()
                

