# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
#import datetime
import numpy as np
import sys


class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30,
            l1=0.0, l2=0.0, epochs=500, eta=0.001,
            alpha=0.0, decrease_const=0.0, shuffle=True,
            minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return 1.0/(1.0 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))
        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)
                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)
                # update weights
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self


def hog(image):
    image = np.reshape(image, (56, 56))
    nwin_x = 7
    nwin_y = 7
    B = 10
    (L, C) = np.shape(image)
    H = np.zeros(shape=(nwin_x*nwin_y*B,1))
    m = np.sqrt(L/2.0)
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C/(nwin_x+1))
    step_y = np.floor(L/(nwin_y+1))
    cont = 0
    hx = np.array([1, 0, -1])
    hy = np.array([-1, 0, 1])
    grad_xr = np.convolve(image.flatten(), hx, mode='same').reshape(56,56)
    grad_yu = np.convolve(image.T.flatten(), hy, mode='same').reshape(56,56).T
    angles = np.arctan2(grad_yu,grad_xr)
    magnit = np.sqrt((grad_yu**2 +grad_xr**2))
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            magnit2 = magnit[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            K = np.shape(v_angles)[0]
            bin = 0
            H2 = np.zeros(shape=(B, 1))
            for ang_lim in np.arange(start=-np.pi+2*np.pi/B,stop=np.pi+2*np.pi/B,step=2*np.pi/B):
                check = v_angles < ang_lim
                v_angles = (v_angles * (~check)) + check * 100
                H2[bin] += np.sum(v_magnit * check)
                bin += 1
            H2 = H2 / (np.linalg.norm(H2)+0.01)
            H[(cont-1)*B:cont*B]=H2
    return H.flatten()


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    data = pkl.load(open('trained_network.pkl', mode='rb'))
    net = NeuralNetMLP(n_output=data['n_out'],
                      n_features=data['n_feat'],
                      n_hidden=data['n_hid'],
                      l2=data['l2'],
                      l1=data['l1'],
                      epochs=data['epo'],
                      eta=data['et'],
                      alpha=data['alph'],
                      decrease_const=sys.float_info.min,
                      shuffle=data['shuff'],
                      minibatches=data['mini'],
                      random_state=data['random_st'])
    net.w1 = data['wu1']
    net.w2 = data['wu2']
    x = np.asarray(x)
    x_temp = np.zeros((x.shape[0], 490))
    for ind in range(x.shape[0]):
        x_temp[ind] = hog(x[ind])
    predictions = net.predict(x_temp)
    predictions = np.reshape(predictions, (len(predictions), 1))
    return predictions


def max_pool(d):
    x = np.array([[(d[w][n] or d[w][n+1] or d[w][n+2] or d[w][n+3] or d[w][n+4] or d[w][n+5] or d[w][n+6]) for n in range(0, d.shape[1], 7)] for w in range(d.shape[0])])
    return x


def divide(data_to_divide, first_part, second_part):
    x1 = np.array([data_to_divide[w, :] for w in range(int(data_to_divide.shape[0]*(first_part/(first_part+second_part))))])
    x2 = np.array([data_to_divide[w, :] for w in range(int(data_to_divide.shape[0]*(first_part/(first_part+second_part))), data_to_divide.shape[0])])
    return x1, x2


def save_object(obj, filename):
    dictio = {'n_out': obj.n_output, 'n_feat': obj.n_features, 'n_hid': obj.n_hidden, 'l2': obj.l2, 'l1': obj.l1,
            'epo': obj.epochs, 'et': obj.eta, 'alph': obj.alpha, 'decr_con': obj.decrease_const,
            'shuff': obj.shuffle, 'mini': obj.minibatches, 'random_st': 1, 'wu1': obj.w1, 'wu2': obj.w2}
    with open(filename, 'wb') as output:
        pkl.dump(dictio, output)


def train():
    data = pkl.load(open('train.pkl', mode='rb'))
    x_train = data[0]
    y_train = data[1]
    x_train = np.asarray(x_train)
    train_x_temp = np.zeros((x_train.shape[0], 490))
    for x in range(x_train.shape[0]):
        train_x_temp[x] = hog(x_train[x])
    nn = NeuralNetMLP(n_output=36,
                      n_features=train_x_temp.shape[1],
                      n_hidden=100,
                      l2=0.1,
                      l1=0.0,
                      epochs=2500,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=sys.float_info.min,
                      shuffle=True,
                      minibatches=32,
                      random_state=1)
    nn.fit(train_x_temp, y_train, print_progress=True)
    save_object(nn, 'trained_network.pkl')
    print('Uczenie zakonczone!')


def test():
    best = 0
    iter = 1
    for i in range(32, 34, 1):
        for j in range(98, 103, 1):
            data = pkl.load(open('train.pkl', mode='rb'))
            #czas = datetime.datetime.now()
            train_x, val_x = divide(data[0], 11, 1)
            #train_x = max_pool(train_x)
            #val_x = max_pool(val_x)
            train_x = np.asarray(train_x)
            val_x = np.asarray(val_x)
            train_x_temp = np.zeros((train_x.shape[0], 490)) #490
            val_x_temp = np.zeros((val_x.shape[0], 490))     #490
            for x in range(train_x.shape[0]):
                train_x_temp[x] = hog(train_x[x])
            for y in range(val_x.shape[0]):
                val_x_temp[y] = hog(val_x[y])
            train_y, val_y = divide(data[1], 11, 1)
            #print("Czas transformacji: ", datetime.datetime.now()-czas)
            #czas2 = datetime.datetime.now()
            #150 hidden(neuronów), 1000 epok, 50 minibatchy
            nn = NeuralNetMLP(n_output=36,
                            n_features=train_x_temp.shape[1],
                            n_hidden=j,
                            l2=0.1,
                            l1=0.0,
                            epochs=1000,
                            eta=0.001,
                            alpha=0.001,
                            decrease_const=sys.float_info.min,
                            shuffle=True,
                            minibatches=i,
                            random_state=1)
            nn.fit(train_x_temp, train_y, print_progress=True)
            y_pred = nn.predict(val_x_temp)
            #print("Czas uczenia i predykcji: ", datetime.datetime.now()-czas2)
            y_pred = np.reshape(y_pred, (len(y_pred), 1))
            acc = np.sum(val_y == y_pred, axis=0) / val_x_temp.shape[0]
            print('iter: %s = %s n_hidden and %s minibatches' % (iter, nn.n_hidden, nn.minibatches))
            if(acc > best):
                print('New record: %.2f%% for %s n_hidden and %s minibatches in %s iter' % ((acc * 100), nn.n_hidden, nn.minibatches, iter))
                best = acc
            iter = iter +1

test()
#train()
#data = pkl.load(open('train.pkl', mode='rb'))
#adat = data[0]
#predict(adat)
