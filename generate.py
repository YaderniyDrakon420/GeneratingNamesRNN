
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

category_lines = {}
all_categories = []

# for filename in findFiles('data/names/*.txt'):
for filename in findFiles('data/cities/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found!')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(
            n_categories + input_size + hidden_size,
            hidden_size
        )

        self.i2o = nn.Linear(
            n_categories + input_size + hidden_size,
            output_size
        )

        self.o2o = nn.Linear(
            hidden_size + output_size,
            output_size
        )

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)

        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)

        output_combined = torch.cat((hidden, output), 1)

        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def categoryTensor(category):
    li = all_categories.index(category)

    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1

    return tensor


# Создание One-Hot матрицы букв
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)

    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1

    return tensor



rnn = RNN(n_letters, 128, n_letters)
rnn.load_state_dict(
    torch.load('rnn_model.pth', weights_only=True)
)
rnn.eval()
print("Модель успешно загружена!")

max_length = 20

def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category)

        input = inputTensor(start_letter)

        hidden = rnn.initHidden()
        output_name = start_letter
        for i in range(max_length):

            output, hidden = rnn(
                category_tensor,
                input[0],
                hidden
            )
            topv, topi = output.topk(1)

            topi = topi[0][0]
            if topi == n_letters - 1:
                break

            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


# samples('Russian', 'RUS')
# samples('German', 'GER')
# samples('Spanish', 'SPA')
# samples('Chinese', 'CHI')

samples('Ukraine', 'KLO')
samples('France', 'PAR')
samples('Germany', 'BER')
samples('Japan', 'TOK')
samples('Italy', 'ROM')