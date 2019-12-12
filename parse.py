from csv import reader
import matplotlib.pyplot as plt
import numpy as np

flickr_data = []
blogcatalog_data = []

flickr_fd = open('flickr_output.csv','r', encoding='utf-8-sig' )
flickr_rd = reader(flickr_fd)

for i in flickr_rd:
    row = list(i)
    row[2] = float(row[2])
    row[3] = float(row[3])
    row[4] = float(row[4])
    flickr_data.append(row)

flickr_fd.close()

blogcatalog_fd = open('blogcatalog_output.csv','r', encoding='utf-8-sig' )
blogcatalog_rd= reader(blogcatalog_fd)

for i in blogcatalog_rd:
    row = list(i)
    row[2] = float(row[2])
    row[3] = float(row[3])
    row[4] = float(row[4])
    blogcatalog_data.append(row)

blogcatalog_fd.close()

fec = {}
fce = {}

e = set()
c = set()

for row in flickr_data:
    e.add(row[0])
    c.add(row[1])

for i in e:
    fec[i] = {}
    for j in c:
        fec[i][j] = []

for row in flickr_data:
    fec[row[0]][row[1]].append(np.array(row[2:]))

# for i in fec:
#     for j in fec[i]:
#         fec[i][j] = np.array(fec[i][j]).T

# for i in fce:
#     for j in fce[i]:
#         fce[i][j] = np.array(fce[i][j]).T

print(e)
print(c)


bec = {}

e = set()
c = set()

for row in blogcatalog_data:
    e.add(row[0])
    c.add(row[1])

for i in e:
    bec[i] = {}
    for j in c:
        bec[i][j] = []

for row in blogcatalog_data:
    bec[row[0]][row[1]].append(np.array(row[2:]))

for i in fec:
    for j in fec[i]:
        fec[i][j] = np.array(fec[i][j]).T

for i in bec:
    for j in bec[i]:
        bec[i][j] = np.array(bec[i][j]).T


i = 0
fig = plt.Figure(figsize=(13, 5))
axs = fig.subplots(2,5)
for emb in ['Node2Vec', 'FastText', 'GloVe', 'CBOW', 'LLE']:
    ax = axs[0][i]
    ax.set_title(emb)
    ax.set(xlabel='test_ratio', ylabel='Macro F1')
    ax.set(ylim=(0,0.25))
    ax.set_xticks(np.arange(0.2, 1, 0.2)) 
    ax.label_outer()
    ax = axs[1][i]
    # ax.set_title(emb)
    ax.set(xlabel='test_ratio', ylabel='Micro F1')
    ax.set(ylim=(0,0.4))
    ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
    ax.label_outer()
    for clas in fec[emb]:
        if emb == 'Node2Vec':
            axs[0][i].plot(fec[emb][clas][0],fec[emb][clas][1], label=clas)
        else:
            axs[0][i].plot(fec[emb][clas][0],fec[emb][clas][1])
        axs[1][i].plot(fec[emb][clas][0],fec[emb][clas][2])
    i += 1

fig.legend(bbox_to_anchor=(0.5,1), loc="upper center", 
                bbox_transform=fig.transFigure, ncol=5)
fig.savefig('flickr2')


i = 0
fig = plt.Figure(figsize=(13, 5))
axs = fig.subplots(2,5)
for emb in ['Node2Vec', 'GloVe', 'FastText', 'HOPE', 'Laplacian Eigenmap']:
    ax = axs[0][i]
    ax.set_title(emb)
    ax.set(xlabel='test_ratio', ylabel='Macro F1')
    ax.set(ylim=(0,0.25))
    ax.set_xticks(np.arange(0.2, 1, 0.2)) 
    ax.label_outer()
    ax = axs[1][i]
    # ax.set_title(emb)
    ax.set(xlabel='test_ratio', ylabel='Micro F1')
    ax.set(ylim=(0,0.4))
    ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
    ax.label_outer()
    for clas in bec[emb]:
        if emb == 'Node2Vec':
            axs[0][i].plot(bec[emb][clas][0],bec[emb][clas][1], label=clas)
        else:
            axs[0][i].plot(bec[emb][clas][0],bec[emb][clas][1])
        axs[1][i].plot(bec[emb][clas][0],bec[emb][clas][2])
    i += 1

fig.legend(bbox_to_anchor=(0.5,1), loc="upper center", 
                bbox_transform=fig.transFigure, ncol=5)
fig.savefig('blogcatalog2')


fig = plt.Figure(figsize=(9.5, 4))
axs = fig.subplots(1,2)
for emb in ['Node2Vec', 'GloVe', 'FastText', 'HOPE', 'Laplacian Eigenmap']:
    clas = 'TopKRanker'
    ax = axs[0]
    ax.set(xlabel='test_ratio', ylabel='Macro F1')
    ax.set(ylim=(0,0.25))
    ax.set_xticks(np.arange(0.2, 1, 0.2)) 
    # ax.label_outer()
    ax.plot(bec[emb][clas][0],bec[emb][clas][1], label=emb)
    ax = axs[1]
    ax.set(xlabel='test_ratio', ylabel='Micro F1')
    ax.set(ylim=(0,0.4))
    ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
    # ax.label_outer()
    ax.plot(bec[emb][clas][0],bec[emb][clas][2])

# fig.suptitle('Comparison of Embedding Models') #, fontsize=16)
fig.legend(bbox_to_anchor=(0.5,1), loc="upper center", 
                bbox_transform=fig.transFigure, ncol=5)
# fig.show()
# input("wait... (press enter)")
# fig.savefig('test2')
fig.savefig('blogcatalog_emb2')


fig = plt.Figure(figsize=(9.5, 4))
axs = fig.subplots(1,2)
for emb in ['Node2Vec', 'FastText', 'GloVe', 'CBOW', 'LLE']:
    clas = 'TopKRanker'
    ax = axs[0]
    ax.set(xlabel='test_ratio', ylabel='Macro F1')
    ax.set(ylim=(0,0.25))
    ax.set_xticks(np.arange(0.2, 1, 0.2)) 
    # ax.label_outer()
    ax.plot(fec[emb][clas][0],fec[emb][clas][1], label=emb)
    ax = axs[1]
    ax.set(xlabel='test_ratio', ylabel='Micro F1')
    ax.set(ylim=(0,0.4))
    ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
    # ax.label_outer()
    ax.plot(fec[emb][clas][0],fec[emb][clas][2])

# fig.suptitle('Comparison of Embedding Models') #, fontsize=16)
fig.legend(bbox_to_anchor=(0.5,1), loc="upper center", 
                bbox_transform=fig.transFigure, ncol=5)
# fig.show()
# input("wait... (press enter)")
fig.savefig('flickr_emb2')


# fig, axs = plt.subplots(2,2)
# for emb in ['Node2Vec', 'FastText', 'GloVe', 'HOPE', 'Laplacian Eigenmap']:
#     clas = 'TopKRanker'
#     ax = axs[0][0]
#     ax.set_title('BlogCatalog')
#     ax.set(xlabel='test_ratio', ylabel='Macro F1')
#     ax.set(ylim=(0,0.25))
#     ax.set_xticks(np.arange(0.2, 1, 0.2)) 
#     ax.label_outer()
#     ax.plot(bec[emb][clas][0],bec[emb][clas][1], label=emb)
#     ax = axs[1][0]
#     ax.set_title('BlogCatalog')
#     ax.set(xlabel='test_ratio', ylabel='Micro F1')
#     ax.set(ylim=(0,0.4))
#     ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
#     ax.label_outer()
#     ax.plot(bec[emb][clas][0],bec[emb][clas][2])

# for emb in ['Node2Vec', 'FastText', 'GloVe', 'CBOW', 'LLE']:
#     clas = 'TopKRanker'
#     ax = axs[0][1]
#     ax.set_title('Flickr')
#     ax.set(xlabel='test_ratio', ylabel='Macro F1')
#     ax.set(ylim=(0,0.25))
#     ax.set_xticks(np.arange(0.2, 1, 0.2)) 
#     ax.label_outer()
#     ax.plot(fec[emb][clas][0],fec[emb][clas][1], label=emb)
#     ax = axs[1][1]
#     ax.set_title('Flickr')
#     ax.set(xlabel='test_ratio', ylabel='Micro F1')
#     ax.set(ylim=(0,0.4))
#     ax.set_xticks(np.arange(0.2, 0.8, 0.2)) 
#     ax.label_outer()
#     ax.plot(fec[emb][clas][0],fec[emb][clas][2])

# fig.suptitle('Comparison of Embedding Models') #, fontsize=16)
# fig.legend(bbox_to_anchor=(0.5,0), loc="lower center", 
#                 bbox_transform=fig.transFigure, ncol=5)
# fig.show()
# fig.savefig('emb')
