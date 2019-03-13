# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

data_split_type = 'cv1'
# jump，
# happy jump，
# squat，
# sad squat，
# throw，
# angry throw，
# stand，
# surprised stand，
# recede，
# fearful recede，
# turn，
# disgusted turn
labels = ['jump', 'happy jump', 'squat', 'sad squat', 'throw', 'angry throw', 'stand', 'surprised stand',
          'recede', 'fearful recede', 'turn', 'disgusted turn']
lmap = [5, -4, 2, -1, -3, 1, 0, 6, 3, -5, 4, -2]
# y_true代表真实的label值 y_pred代表预测得到的lavel值
y_true = np.load('./%s_mfigs/y_true.npy' % data_split_type)

y_pred = np.load('./%s_mfigs/y_pred.npy' % data_split_type)

tick_marks = np.array(range(len(labels))) + 0.5

c_map = plt.cm.get_cmap('Blues')


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=c_map):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
a = np.zeros((12, 12))
for r, row in enumerate(cm_normalized):
    for c, num in enumerate(row):
        a[lmap[r]][lmap[c]] = num
cm_normalized = a
print(cm_normalized)
print(cm_normalized.shape)
plt.figure(figsize=(12, 10), dpi=512)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.5:
        color = "white"
    else:
        color = "black"
    plt.text(x_val, y_val, "%0.2f" % (c,), color=color, fontsize=11, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='normalized confusion matrix on %s of our model' % data_split_type)
# show confusion matrix
plt.savefig('./%s_mfigs/%s_confusion_matrix_sorted.svg' % (data_split_type, data_split_type), format='svg')
plt.show()
print(data_split_type)
