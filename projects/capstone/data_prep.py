

#The below information is from B H's Kernel. (https://www.kaggle.com/zero92/tmdb-prediction)

def complete_missing_data(data):

    train = data['train']

    train.loc[train['id'] == 16,'revenue'] = 192864
    train.loc[train['id'] == 90,'budget'] = 30000000
    train.loc[train['id'] == 118,'budget'] = 60000000
    train.loc[train['id'] == 149,'budget'] = 18000000
    train.loc[train['id'] == 313,'revenue'] = 12000000
    train.loc[train['id'] == 451,'revenue'] = 12000000
    train.loc[train['id'] == 464,'budget'] = 20000000
    train.loc[train['id'] == 470,'budget'] = 13000000
    train.loc[train['id'] == 513,'budget'] = 930000
    train.loc[train['id'] == 797,'budget'] = 8000000
    train.loc[train['id'] == 819,'budget'] = 90000000
    train.loc[train['id'] == 850,'budget'] = 90000000
    train.loc[train['id'] == 1007,'budget'] = 2
    train.loc[train['id'] == 1112,'budget'] = 7500000
    train.loc[train['id'] == 1131,'budget'] = 4300000
    train.loc[train['id'] == 1359,'budget'] = 10000000
    train.loc[train['id'] == 1542,'budget'] = 1
    train.loc[train['id'] == 1570,'budget'] = 15800000
    train.loc[train['id'] == 1571,'budget'] = 4000000
    train.loc[train['id'] == 1714,'budget'] = 46000000
    train.loc[train['id'] == 1721,'budget'] = 17500000
    train.loc[train['id'] == 1865,'revenue'] = 25000000
    train.loc[train['id'] == 1885,'budget'] = 12
    train.loc[train['id'] == 2091,'budget'] = 10
    train.loc[train['id'] == 2268,'budget'] = 17500000
    train.loc[train['id'] == 2491,'budget'] = 6
    train.loc[train['id'] == 2602,'budget'] = 31000000
    train.loc[train['id'] == 2612,'budget'] = 15000000
    train.loc[train['id'] == 2696,'budget'] = 10000000
    train.loc[train['id'] == 2801,'budget'] = 10000000
    train.loc[train['id'] == 335,'budget'] = 2
    train.loc[train['id'] == 348,'budget'] = 12
    train.loc[train['id'] == 470,'budget'] = 13000000
    train.loc[train['id'] == 513,'budget'] = 1100000
    train.loc[train['id'] == 640,'budget'] = 6
    train.loc[train['id'] == 696,'budget'] = 1
    train.loc[train['id'] == 797,'budget'] = 8000000
    train.loc[train['id'] == 850,'budget'] = 1500000
    train.loc[train['id'] == 1199,'budget'] = 5
    train.loc[train['id'] == 1282,'budget'] = 9
    train.loc[train['id'] == 1347,'budget'] = 1
    train.loc[train['id'] == 1755,'budget'] = 2
    train.loc[train['id'] == 1801,'budget'] = 5
    train.loc[train['id'] == 1918,'budget'] = 592
    train.loc[train['id'] == 2033,'budget'] = 4
    train.loc[train['id'] == 2118,'budget'] = 344
    train.loc[train['id'] == 2252,'budget'] = 130
    train.loc[train['id'] == 2256,'budget'] = 1
    train.loc[train['id'] == 2696,'budget'] = 10000000

    test = data['test']

    test.loc[test['id'] == 3033,'budget'] = 250
    test.loc[test['id'] == 3051,'budget'] = 50
    test.loc[test['id'] == 3084,'budget'] = 337
    test.loc[test['id'] == 3224,'budget'] = 4
    test.loc[test['id'] == 3594,'budget'] = 25
    test.loc[test['id'] == 3619,'budget'] = 500
    test.loc[test['id'] == 3831,'budget'] = 3
    test.loc[test['id'] == 3935,'budget'] = 500
    test.loc[test['id'] == 4049,'budget'] = 995946
    test.loc[test['id'] == 4424,'budget'] = 3
    test.loc[test['id'] == 4460,'budget'] = 8
    test.loc[test['id'] == 4555,'budget'] = 1200000
    test.loc[test['id'] == 4624,'budget'] = 30
    test.loc[test['id'] == 4645,'budget'] = 500
    test.loc[test['id'] == 4709,'budget'] = 450
    test.loc[test['id'] == 4839,'budget'] = 7
    test.loc[test['id'] == 3125,'budget'] = 25
    test.loc[test['id'] == 3142,'budget'] = 1
    test.loc[test['id'] == 3201,'budget'] = 450
    test.loc[test['id'] == 3222,'budget'] = 6
    test.loc[test['id'] == 3545,'budget'] = 38
    test.loc[test['id'] == 3670,'budget'] = 18
    test.loc[test['id'] == 3792,'budget'] = 19
    test.loc[test['id'] == 3881,'budget'] = 7
    test.loc[test['id'] == 3969,'budget'] = 400
    test.loc[test['id'] == 4196,'budget'] = 6
    test.loc[test['id'] == 4221,'budget'] = 11
    test.loc[test['id'] == 4222,'budget'] = 500
    test.loc[test['id'] == 4285,'budget'] = 11
    test.loc[test['id'] == 4319,'budget'] = 1
    test.loc[test['id'] == 4639,'budget'] = 10
    test.loc[test['id'] == 4719,'budget'] = 45
    test.loc[test['id'] == 4822,'budget'] = 22
    test.loc[test['id'] == 4829,'budget'] = 20
    test.loc[test['id'] == 4969,'budget'] = 20
    test.loc[test['id'] == 5021,'budget'] = 40
    test.loc[test['id'] == 5035,'budget'] = 1
    test.loc[test['id'] == 5063,'budget'] = 14
    test.loc[test['id'] == 5119,'budget'] = 2
    test.loc[test['id'] == 5214,'budget'] = 30
    test.loc[test['id'] == 5221,'budget'] = 50
    test.loc[test['id'] == 4903,'budget'] = 15
    test.loc[test['id'] == 4983,'budget'] = 3
    test.loc[test['id'] == 5102,'budget'] = 28
    test.loc[test['id'] == 5217,'budget'] = 75
    test.loc[test['id'] == 5224,'budget'] = 3
    test.loc[test['id'] == 5469,'budget'] = 20
    test.loc[test['id'] == 5840,'budget'] = 1
    test.loc[test['id'] == 5960,'budget'] = 30
    test.loc[test['id'] == 6506,'budget'] = 11
    test.loc[test['id'] == 6553,'budget'] = 280
    test.loc[test['id'] == 6561,'budget'] = 7
    test.loc[test['id'] == 6582,'budget'] = 218
    test.loc[test['id'] == 6638,'budget'] = 5
    test.loc[test['id'] == 6749,'budget'] = 8
    test.loc[test['id'] == 6759,'budget'] = 50
    test.loc[test['id'] == 6856,'budget'] = 10
    test.loc[test['id'] == 6858,'budget'] =  100
    test.loc[test['id'] == 6876,'budget'] =  250
    test.loc[test['id'] == 6972,'budget'] = 1
    test.loc[test['id'] == 7079,'budget'] = 8000000
    test.loc[test['id'] == 7150,'budget'] = 118
    test.loc[test['id'] == 6506,'budget'] = 118
    test.loc[test['id'] == 7225,'budget'] = 6
    test.loc[test['id'] == 7231,'budget'] = 85
    test.loc[test['id'] == 5222,'budget'] = 5
    test.loc[test['id'] == 5322,'budget'] = 90
    test.loc[test['id'] == 5350,'budget'] = 70
    test.loc[test['id'] == 5378,'budget'] = 10
    test.loc[test['id'] == 5545,'budget'] = 80
    test.loc[test['id'] == 5810,'budget'] = 8
    test.loc[test['id'] == 5926,'budget'] = 300
    test.loc[test['id'] == 5927,'budget'] = 4
    test.loc[test['id'] == 5986,'budget'] = 1
    test.loc[test['id'] == 6053,'budget'] = 20
    test.loc[test['id'] == 6104,'budget'] = 1
    test.loc[test['id'] == 6130,'budget'] = 30
    test.loc[test['id'] == 6301,'budget'] = 150
    test.loc[test['id'] == 6276,'budget'] = 100
    test.loc[test['id'] == 6473,'budget'] = 100
    test.loc[test['id'] == 6842,'budget'] = 30
    
    
    test.loc[test['id'] == 3829,'release_date'] = '05/01/2000'

    
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d