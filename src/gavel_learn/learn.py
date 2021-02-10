import torch
from gavel.logic import logic as fol
from gavel.logic import problem
from gavel.dialects.base.compiler import Compiler
from gavel_learn.simplifier import MapExtractor, MaskCompiler, MaskedElement, EncodedElement
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from sklearn import metrics


class FormulaNet(torch.nn.Module, Compiler):

    def __init__(self):
        super().__init__()
        self._leaf_factor = 3
        self._leaf_offset = 5
        self.length = 50
        self.argument_limit = 5
        self.max_embeddings = 100
        self.attention = Attention(self.length)
        self.output_size = self._leaf_factor*self.length + self._leaf_offset
        self.unary_formula = torch.nn.Linear(self.length, self.length)
        self.final = torch.nn.Linear(self.length, self.output_size)
        self.existential_quant = torch.nn.Linear(self.length, self.length)
        self.universal_quant = torch.nn.Linear(self.length, self.length)
        self.binary_formula = torch.nn.Linear(3*self.length, self.length)
        self.predicate_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.functor_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.leaf_net = torch.nn.Linear(self._leaf_factor * self.length + self._leaf_offset, self.length)

        self._ce = torch.nn.Embedding(self.max_embeddings, self.length)
        self._fe = torch.nn.Embedding(self.max_embeddings, self.length)
        self._pe = torch.nn.Embedding(self.max_embeddings, self.length)
        self._oe = torch.nn.Embedding(self.max_embeddings, self.length)
        self._ve = torch.nn.Embedding(self.max_embeddings, self.length)

        self.embeddings = dict(
            constants=self._ce,
            functors = self._fe,
            predicates = self._pe,
            binary_operators = self._oe,
            variables = self._ve
        )

        self._null = torch.autograd.Variable(torch.randn(self.length))
        self._masked = torch.zeros(self._leaf_factor * self.length + self._leaf_offset)


    """def _generate_encodings(self, index):
        identifiers = torch.stack(
            [torch.ones(self._leaf_factor * self.length + 1) if i == index else torch.zeros(self._leaf_factor * self.length + 1) for
             i in range(self._leaf_offset)]).T
        values = torch.eye(self._leaf_factor * self.length)
        overflow = torch.zeros(self._leaf_factor * self.length)
        values = torch.cat((values, torch.reshape(overflow,(1,-1))))
        return torch.cat((identifiers,values), dim=1).to(self.device)"""

    def visit_encoded_element(self, elem:EncodedElement):
        if elem.value < self.max_embeddings:
            return self.embeddings[elem.kind](torch.tensor(elem.value))
        else:
            return self._null

    def forward(self, input: fol.LogicExpression):
        return self.final(self.visit(input))

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        return torch.relu(self.unary_formula(self.visit(formula.formula, **kwargs)))

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        if formula.quantifier == fol.Quantifier.EXISTENTIAL:
            return torch.relu(self.existential_quant(self.visit(formula.formula)))
        else:
            return torch.relu(self.universal_quant(self.visit(formula.formula)))

    def visit_annotated_formula(self, anno: problem.AnnotatedFormula, **kwargs):
        return self.visit(anno.formula)

    def visit_binary_formula(self, formula: fol.BinaryFormula):
        ops = [self.visit(formula.left),self.visit(formula.operator), self.visit(formula.right)]
        return torch.relu(self.binary_formula(torch.cat(ops, dim=-1))) + self.attention(torch.stack(ops))

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        arguments = [self.visit(expression.functor)]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(self._null)
        return torch.relu(self.functor_formula(torch.cat(arguments, dim=-1))) + self.attention(torch.stack(arguments))

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        arguments = [self.visit(expression.predicate)]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(self._null)
        return torch.relu(self.predicate_formula(torch.cat(arguments, dim=-1))) + self.attention(torch.stack(arguments))

    def visit_masked(self, masked: MaskedElement):
        return self._masked


class Attention(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Linear(n,1)

    def forward(self, input):
        filter = torch.softmax(self.weights(input), dim=0).t()
        return torch.sum(torch.matmul(filter, input), dim=0)


class PremiseSelector(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.formula_net = FormulaNet()
        self.encoder = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.conjecture_squash = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.scoring = torch.nn.Linear(self.formula_net.output_size*3, self.formula_net.output_size)
        self.softmax = torch.nn.Softmax(-1)
        self.decoder = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.final = torch.nn.Linear(self.formula_net.output_size, 1)

    def attention_loop(self, premise_stack, hidden, conj, thought=None):
        ratings = self.softmax(self.scoring(torch.cat((premise_stack, conj, hidden), dim=-1)))
        context = torch.sum(torch.mul(premise_stack, ratings), dim=0, keepdim=True)
        if thought is None:
            thought = context.unsqueeze(0)
        else:
            thought = torch.cat((thought, context))
        o, hidden = self.decoder.forward(thought)
        return torch.sigmoid(self.final(o[-1])).squeeze(-1), hidden

    def forward(self, premises, conjectures):
        self.formula_net.prepare(premises, conjectures)
        premise_stack = torch.stack([self.formula_net.forward(p) for p in premises]).unsqueeze(1)
        num_prem = premise_stack.size()[0]
        hidden = premise_stack[-1]
        conj = torch.relu(self.conjecture_squash.forward((torch.stack([self.formula_net.forward(c) for c in conjectures]).unsqueeze(1)))[0].expand(num_prem,1,self.formula_net.output_size))
        thought = None
        output = []
        for i in range(num_prem):
            hidden = hidden.expand(num_prem, 1, self.formula_net.output_size)
            o, hidden = self.attention_loop(premise_stack, hidden, conj, thought)
            output.append(o)
        return torch.cat(output)


class PremiseSelectorGRU(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.formula_net = FormulaNet()
        self.conjecture_squash = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.gru = torch.nn.GRU(self.formula_net.output_size*2, self.formula_net.output_size, bidirectional=True)
        self.final = torch.nn.Linear(self.formula_net.output_size * 2, 1)
        self._nothing = torch.autograd.Variable(torch.rand((1,self.formula_net.output_size)))

    def forward(self, data):
        premise_stack = torch.nn.utils.rnn.pad_sequence([torch.stack([self.formula_net.forward(p) for p in batch[0]]) for batch in data])
        conjecture_stack = torch.nn.utils.rnn.pad_sequence(
            [torch.stack([self.formula_net.forward(c) for c in batch[1]]) if batch[1] else self._nothing for batch in
             data])
        num_prem = premise_stack.size()[0]
        batches = premise_stack.size()[1]
        conj = torch.relu(self.conjecture_squash.forward(conjecture_stack)[0][-1].expand(num_prem,batches,self.formula_net.output_size))
        o, h = self.gru(torch.cat((premise_stack, conj), dim=-1))
        return self.final(o).squeeze(-1)


def train_selection(gen, gen_val):
    net = PremiseSelectorGRU()
    optimizer = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    learning_curve = []
    min_loss = None
    for epoch in range(100):
        curr_loss, f1 = execute(loss, gen, optimizer, net)
        val_loss, val_f1 = execute(loss, gen_val, optimizer, net)
        print(epoch, ",".join(("{0:.5f}".format(x) for x in [curr_loss, f1, val_loss, val_f1])))
        if min_loss is None or min_loss > val_loss:
            min_loss = val_loss
            torch.save(net.state_dict(), "model.{i}.best".format(i=epoch))
        learning_curve.append(curr_loss)

    plt.plot(np.array(learning_curve), 'r')
    plt.savefig("curve.png")
    with open("curve.json", "w") as o:
        o.write(str(learning_curve))
    torch.save(net.state_dict(), "model.last")
    print('Finished Training')


def execute(loss, gen, optimizer, net):
    i = 0
    batch_loss = 0
    batchnumber = 0
    running_f1 = 0
    for data, labels in gen():
        batchnumber += 1
        optimizer.zero_grad()
        predictions = net.forward(data)
        ls = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels])
        l = loss(predictions, ls)
        running_f1 += metrics.f1_score(ls>0.5, predictions>0.5)
        l.backward()
        batch_loss += l.item()
        optimizer.step()
        i += len(data)
    return batch_loss / batchnumber, running_f1 / batchnumber
