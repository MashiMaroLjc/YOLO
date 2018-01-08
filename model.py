# coding:utf-8

from mxnet.gluon import nn
from mxnet import gluon
from mxnet.initializer import Xavier
from mxnet import gpu, cpu
from mxnet import nd
import pickle
from loss import *
from mxnet import autograd
from utils import *
from time import time
from mxnet import optimizer
from dataloader import BaseDataLoader




class Yolo(gluon.Block):
    def __init__(self, layer_num, class_num, class_name,s=7, b=2, **kwargs):
        """

        :param layer_num:
        :param class_num:
        :param s:
        :param b:
        :param verbose:
        :param kwargs:
        """
        super(Yolo, self).__init__(**kwargs)
        self._s = s
        self._b = b
        self._class_num = class_num
        self._layer_num = layer_num
        self._class_name = class_name
        assert len(self._class_name) == self._class_num
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.out.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.05)),
            self.out.add(nn.MaxPool2D(2))

            self.out.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.1)),
            self.out.add(nn.MaxPool2D(2))

            self.out.add(nn.Conv2D(16, kernel_size=1, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.05)),
            self.out.add(nn.MaxPool2D(2))

            self.out.add(nn.Conv2D(16, kernel_size=3, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.05)),
            self.out.add(nn.MaxPool2D(2))

            self.out.add(nn.Conv2D(16, kernel_size=1, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.05)),
            self.out.add(nn.Conv2D(16, kernel_size=3, strides=1, padding=1)),
            self.out.add(nn.BatchNorm()),
            self.out.add(nn.LeakyReLU(0.05)),
            self.out.add(nn.Flatten())

            self.out.add(nn.Dense(128))
            self.out.add(nn.LeakyReLU(0.05))
            self.out.add(nn.Dense(self._s * self._s * (self._b * 5 + class_num)))

    @property
    def s(self):
        return self._s

    @property
    def b(self):
        return self._b

    @property
    def class_num(self):
        return self._class_num

    @property
    def class_names(self):
        return self._class_name
    @property
    def layer_num(self):
        return self._layer_num

    def forward(self, x):
        return self.out(x)


def train(params, loader, model=None):
    epoch = params.get('epoch', 10)
    verbose = params.get("verbose", True)
    batch_size = params.get("batch_size", 32)
    if model is None:
        class_name = params["class_name"]
        layer_num = params.get("layer_num", 5)
        class_num = params.get("class_num", 3)
        s = params.get("s", 4)
        b = params.get("b", 2)
        yolo = Yolo(layer_num, class_num, s=s, b=b,class_name=class_name)

        yolo.initialize(init=Xavier(magnitude=0.02))
    else:
        print("model load finish")
        layer_num = model.layer_num
        class_num = model.class_num
        s = model.s
        b = model.b
        yolo = model
    if verbose:
        print("train params: \n\tepoch:%d \n\tlayer_num:%d \n\tclass_num:%d  \n\ts:%d  \n\tb:%d" % \
              (epoch, layer_num, class_num, s, b))

    ngd = optimizer.SGD(momentum=0.7,learning_rate=0.005)
    trainer = gluon.Trainer(yolo.collect_params(), ngd)

    for ep in range(epoch):
        loader.reset()
        mean_loss = 0
        t1 = time()
        for i, batch in enumerate(loader):
            x = batch.data[0]
            y = batch.label[0].reshape((-1, 5))
            y = translate_y(y, yolo.s, yolo.b, yolo.class_num)
            y = nd.array(y)
            with autograd.record():
                loss_func = TotalLoss(s=s, c=class_num, b=b)
                ypre = yolo(x)  # (32,output_dim)
                loss = nd.mean(loss_func(ypre, y))
                mean_loss += loss.asscalar()
            loss.backward()
            trainer.step(batch_size)
        t2 = time()
        if verbose:
            print("epoch:%d/%d  loss:%.5f  time:%4f" % (
                ep + 1, epoch, mean_loss/32, t2 - t1),
                  flush=True)

        print()
    return yolo


def train2(params, loader: BaseDataLoader, model=None):
    epoch = params.get('epoch', 10)
    verbose = params.get("verbose", True)
    batch_size = params.get("batch_size", 32)
    if model is None:
        layer_num = params.get("layer_num", 5)
        class_num = params.get("class_num", 3)
        s = params.get("s", 4)
        b = params.get("b", 2)
        yolo = Yolo(layer_num, class_num, s=s, b=b)

        yolo.initialize(init=Xavier(magnitude=0.02))
    else:
        print("model load finish")
        layer_num = model.layer_num
        class_num = model.class_num
        s = model.s
        b = model.b
        yolo = model
    if verbose:
        print("train params: \n\tepoch:%d \n\tlayer_num:%d \n\tclass_num:%d  \n\ts:%d  \n\tb:%d" % \
              (epoch, layer_num, class_num, s, b))

    ngd = optimizer.SGD(momentum=0.7,learning_rate=0.0025)
    trainer = gluon.Trainer(yolo.collect_params(), ngd)

    for ep in range(epoch):
        loss = 0
        all_batch = int(loader.data_number() / batch_size)
        t1 = time()
        for _ in range(all_batch):
            x, y = loader.next_batch(batch_size)
            with autograd.record():
                loss_func = TotalLoss(s=s, c=class_num, b=b)
                ypre = yolo(x)  # (32,output_dim)
                loss = nd.mean(loss_func(ypre, y))
            loss.backward()
            trainer.step(batch_size)

        t2 = time()
        if verbose:
            print("epoch:%d/%d  loss:%.5f  time:%4f" % (
                ep + 1, epoch, loss.asscalar(), t2 - t1),
                  flush=True)

    return yolo


def save(model: Yolo, pre="yolo"):
    pickle.dump(model, open(pre + ".pick", "wb"))
    model.save_params(pre + ".params")


def load(model_file, params_file):
    if not model_file.endswith(".pick"):
        model_file += ".pick"

    model = pickle.load(open(model_file, "rb"))
    if isinstance(model, Yolo):
        if not params_file.endswith(".params"):
            params_file += ".params"
        model.load_params(params_file, ctx=cpu(0))
    else:
        raise ValueError("What is you want you load.It should be yolo model")
    return model


def predict(yolo:Yolo,x,threshold=0.5):
    """
    return label ,C,location
    :param yolo:
    :return:
    """
    assert  len(x)==1,"Only One image for now"
    ypre = yolo(x)
    label, preds, location = deal_output(ypre, yolo.s, b=yolo.b, c=yolo.class_num)
    indexs = []
    for i,c in enumerate(preds[0]):
        if c > threshold:
            indexs.append(i)
    class_names = []
    C_list  =[]
    bos_list = []
    for index in indexs:
        label_index = int(index / 2)
        location_offect = int(index % 2)
        class_index = nd.argmax(label[0][label_index], axis=0)
        C = preds[0][index]
        locat = location[0][label_index][location_offect]
        C_list.append(C.asscalar())
        #######traslate the name
        label_name = yolo.class_names
        text = label_name[int(class_index.asscalar()) ]
        class_names.append(text)
        ###traslate the locat
        x, y, w, h = locat
        w, h = nd.power(w, 2), nd.power(h, 2)
        ceil = 1 / 4
        row = int(label_index / 4)
        columns = label_index % 4
        x_center = columns * ceil + x
        y_center = row * ceil + y
        x_min, y_min, x_max, y_max = x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w, y_center + 0.5 * h
        box = nd.concatenate([x_min, y_min, x_max, y_max], axis=0) * 256
        bos_list.append(box.asnumpy())
        return class_names,C_list,bos_list