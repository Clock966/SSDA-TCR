import random
with open('/home/ll/Documents/zjh/sv-adapt/plot_tsne_data_new/wav_for_plot.scp','r')as f:
    lines = f.readlines()
select_list = []
post_list = []
nage_list = []

while(len(select_list)<8000):
    a,b = random.sample(lines,2)
    id_a,path_a = a.rsplit()[0],a.rsplit()[-1]
    #id_a = wav_a.rsplit('/')[0]
    id_b, path_b = b.rsplit()[0], b.rsplit()[-1]
    #id_b = wav_b.rsplit('/')[0]

    if id_a==id_b and len(post_list)<4000:
        new_line1 = path_a+' '+path_b+' '+ 'target'+'\n'
        new_line2 = path_b + ' ' + path_a + ' ' + 'target' + '\n'
        if new_line1 not in post_list or new_line2 not in post_list:
            post_list.append(new_line1)
            select_list.append(new_line1)
    if id_a!=id_b and len(nage_list)<4000:
        new_line1 = path_a + ' ' + path_b + ' ' + 'nontarget' + '\n'
        new_line2 = path_b + ' ' + path_a + ' ' + 'nontarget' + '\n'
        if new_line1 not in nage_list or new_line2 not in nage_list:
            nage_list.append(new_line1)
            select_list.append(new_line1)
with open('/home/ll/Documents/zjh/sv-adapt/plot_tsne_data_new/wav_for_plot.trail','w')as f:
    for i in select_list:
        f.write(i)
