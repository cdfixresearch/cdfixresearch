import torch
from torch.nn import Dropout, GRU, Linear, Module, Embedding
from torch.autograd import Variable as Var
import torch.nn.functional as F


device = torch.device("cpu")
MAX_LENGTH = 20


class TreeSummarize(Module):
    def __init__(self, in_dim, mem_dim, dropout1):
        super(TreeSummarize, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = Linear(self.in_dim, self.mem_dim)
        self.fh = Linear(self.mem_dim, self.mem_dim)
        self.H = []
        self.drop = Dropout(dropout1)

    def node_forward(self, inputs, child_c, child_h):
        inputs = torch.unsqueeze(inputs, 0)
        child_h_sum = torch.sum(child_h, dim=0)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        self.H.append(h)
        return c, h

    def forward(self, data, record):
        tree = data[0]
        inputs = data[1]
        for idx in range(tree.num_children):
            _, record = self.forward([tree.children[idx], inputs], record)

        if tree.num_children == 0:
            child_c = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.id], child_c, child_h)
        record[0].append(tree.id)
        record[1].append(tree.state)
        return tree.state, record


class Encoder(Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = Embedding(input_size, hidden_size)
        self.gru = GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output.float(), hidden.float())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoder_(Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder_, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = Embedding(self.output_size, self.hidden_size)
        self.attn = Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = Dropout(self.dropout_p)
        self.gru = GRU(self.hidden_size, self.hidden_size)
        self.out = Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = input.view(1, 1, -1)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0].float(), hidden[0].float()), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output.float()).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoder(Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_r=0.5, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.dropout_r = dropout_r

        self.attn = Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = Dropout(self.dropout_p)
        self.tree = TreeSummarize(self.hidden_size, self.hidden_size, self.dropout_r)
        self.out = Linear(self.hidden_size, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        embedded = input_data.view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.tree(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class CDFix(torch.nn.Module):
    def __init__(self, h_size, feature_representation_size, drop_out_rate, max_node_amount):
        super(CDFix, self).__init__()
        self.h_size = h_size
        self.max_node_amount = max_node_amount
        self.feature_representation_size = feature_representation_size
        self.drop_out_rate = drop_out_rate
        self.encoder_1 = TreeSummarize(self.feature_representation_size, self.h_size, self.drop_out_rate)
        self.encoder_2 = TreeSummarize(self.feature_representation_size, self.h_size, self.drop_out_rate)
        self.decoder_1 = AttnDecoder(self.h_size, self.h_size, self.max_node_amount[0])
        self.decoder_2 = AttnDecoder(self.h_size, self.h_size, self.max_node_amount[1])

    def forward(self, data):
        output_1, hidden_1 = self.encoder_1(data[0])
        output_2, hidden_2 = self.encoder_2(data[1])
        _, record_1, _ = self.decoder_1(data[0], hidden_1, output_1)
        _, record_2, _ = self.decoder_2(data[1], hidden_2, output_2)
        return record_1, record_2



