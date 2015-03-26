from .._pyirt import irt


#LAST7_data = utl.loader.from_matrix_to_list('data/LAST7.txt',sep='\t',is_uid=True)
src_handle = open('data/sim_data_simple.txt','r')

# load file handle
print('Load file handle.')
item_param,user_param = irt(src_handle)
src_handle.close()

# load tuples
print('Load tuple data.')
src_handle = open('data/sim_data_simple.txt','r')
src_data = []
for line in src_handle:
    if line == '':
        continue
    uidstr, eidstr, atagstr = line.strip().split(',')
    src_data.append((int(uidstr),int(eidstr),int(atagstr)))

item_param,user_param = irt(src_data)

# mount the damn tmp folder into memory
print('Use bdm')
item_param,user_param = irt(src_data, mode = 'bdm')

print('Use bdm by RAM disk')
item_param,user_param = irt(src_data, mode = 'bdm',is_mount=True, user_name="junchen")


#utl.loader.parse_item_paramer(item_param, output_file = 'data/sim_est.txt')

