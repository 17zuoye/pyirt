
from .._pyirt import irt

src_handle = open('data/big_data.txt','r')
src_data = []
for line in src_handle:
    if line == '':
        continue
    uidstr, eidstr, atagstr = line.strip().split('\t')
    src_data.append((int(uidstr),int(eidstr),int(atagstr)))
src_handle.close()

item_param,user_param = irt(src_data[0:1000000])

