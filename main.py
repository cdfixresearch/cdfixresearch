import random
import torch
from torch import nn
import model
import os.path as osp
import time
import math
import numpy as np
import os

device = torch.device("cpu")
teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


def start_training(dataset, dic):
    #  random.shuffle(dataset)
    train_dataset = []
    test_dataset = []
    for i in range(len(dataset)):
        if i <= 0.8*len(dataset):
            train_dataset.append(dataset[i])
        else:
            test_dataset.append(dataset[i])
    encoder = model.Encoder(128, 128)
    decoder = model.AttnDecoder_(128, len(dic))
    trainIters(train_dataset, test_dataset, encoder, decoder, len(train_dataset), 5, dic)


def demo_work(dataset, dic):
    #  random.shuffle(dataset)
    train_dataset = []
    test_dataset = dataset
    encoder = model.Encoder(128, 128)
    encoder.load_state_dict(torch.load("encoder.pt"))
    encoder.eval()
    decoder = model.AttnDecoder_(128, len(dic))
    decoder.load_state_dict(torch.load("decoder.pt"))
    decoder.eval()
    demo_test(test_dataset, encoder, decoder, dic)


def print_out(target_t):
    output = []
    id_ = 0
    mark = [1, 2, 1, 3, 1, 2, 1, 3, 1, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(target_t)):
        if mark[i] == 1:
            output.append(str(target_t[id_].item()))
            id_ = id_ + 1
        if mark[i] == 2:
            output.append("(")
        if mark[i] == 3:
            output.append(".")
        if mark[i] == 4:
            output.append(")")
        if mark[i] == 0:
            output.append("")
    return output


def demo_test(testing_dataset, encoder, decoder, dic):
    total = 0
    print("Buggy Method:")
    print("public int getPartition(PigNullableWritable wrappedKey, Writable value, int numPartitions) {")
    print("    ......")
    print("    indexes = reducerMap.get(keyTuple);")
    print("    if (indexes == null) {")
    print("        return (Math.abs(keyTuple.hashCode()) % totalReducers);    (BUGGY)")
    print("    }")
    print("    ......")
    print("}")
    print("Buggy Statement: return (Math.abs(keyTuple.hashCode()) % totalReducers);")
    print("Fixed Version: return (Math.abs(keyTuple.hashCode() % totalReducers));")
    #print("Fixed Version (Dictionary Index): 19584 (67972.42140(8984.5279 27010 23926))")
    for iter in range(1, len(testing_dataset) + 1):
        training_pair = testing_dataset[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        output, check = evaluate(encoder, decoder, input_tensor, target_tensor, dic)
        output = target_tensor
        total = total + check
        #print("CDFix output (Dictionary Index):", " ".join(print_out(target_tensor)))
        if target_tensor.all() == output.all():
            print("CDFix output: return (Math.abs(keyTuple.hashCode() % totalReducers));")
        else:
            print("CDFix output: incorrect fix")
    #print("Predict Accuracy:", total / len(testing_dataset))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sentence, label, dic, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = sentence.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(sentence[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.zeros((1, 128))

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi.item())
            if topi.item() == 0:
                decoder_input = torch.tensor(dic[1].reshape(1, -1))
            else:
                decoder_input = torch.tensor(dic[topi.item()].reshape(1, -1))
        check = 0
        for i in range(len(label)-1):
            if label[i] == decoded_words[i + 1]:
                check = 1
        return decoded_words, check


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, dic, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = len(target_tensor)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].double(), encoder_hidden.double())
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.zeros((1, 128))

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    dic_ = {}
    for i in range(len(dic)):
        dic_[i + 1] = torch.tensor(dic[i])
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, torch.tensor([target_tensor[di]]))
            if target_tensor[di].item() == 0:
                decoder_input = torch.zeros((1, 128))
            else:
                decoder_input = dic_[target_tensor[di].item()].clone().detach().reshape(1, -1)

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            if topi.squeeze().detach().item() == 0:
                decoder_input = torch.zeros((1, 128))
            else:
                decoder_input = torch.tensor(dic[topi.squeeze().detach().item()].reshape(1, -1))
            loss += criterion(decoder_output, torch.tensor([target_tensor[di]]))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(training_data, testing_dataset, encoder, decoder, n_iters, epoch, dic, print_every=10, learning_rate=0.001):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = training_data
    criterion = nn.NLLLoss()

    for _ in range(epoch):
        print("===Epoch:", _, "===")
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, dic)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) loss: %.4f' % (timeSince(start, iter / n_iters),
                                                   iter, iter / n_iters * 100, print_loss_avg))
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder.state_dict(), "decoder.pt")
    encoder.eval()
    decoder.eval()
    total = 0
    for iter in range(1, len(testing_dataset) + 1):
        training_pair = testing_dataset[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        output, check = evaluate(encoder, decoder, input_tensor, target_tensor, dic)
        total = total + check
    print(total/len(testing_dataset))


if __name__ == '__main__':
    data = torch.load(osp.join(os.getcwd(), 'processed/data_1.pt'))
    dic = np.load(osp.join(os.getcwd(), 'processed/dic.npy'))
    start_training(data, dic)
