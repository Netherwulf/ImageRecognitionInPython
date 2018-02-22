import pickle
import datetime
import numpy as np
from predict import predict, divide

data = pickle.load(open('train.pkl', mode='rb'))
adat = data[0]
adat, _ = divide(adat, 1, 11)
val_y, _ = divide(data[1], 1, 11)
czas = datetime.datetime.now()
pred = predict(adat)
print("Czas predicta 2500 obrazow: ", datetime.datetime.now()-czas)
acc = np.sum(val_y == pred, axis=0) / adat.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))
