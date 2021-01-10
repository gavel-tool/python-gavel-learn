import torch
from gavel.logic import logic as fol
from gavel.logic import problem
from gavel.dialects.base.compiler import Compiler
from gavel_learn.simplifier import MapExtractor, MaskCompiler, MaskedElement, EncodedElement
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu:0")

class FormulaNet(torch.nn.Module, Compiler):

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self._leaf_factor = 3
        self._leaf_offset = 5
        self.length = 50
        self.argument_limit = 5
        self.max_embeddings = 100
        self.output_size = self._leaf_factor*self.length + self._leaf_offset
        self.unary_formula = torch.nn.Linear(self.length, self.length)
        self.final = torch.nn.Linear(self.length, self.output_size)
        self.existential_quant = torch.nn.Linear(self.length, self.length)
        self.universal_quant = torch.nn.Linear(self.length, self.length)
        self.binary_formula = torch.nn.Linear(3*self.length, self.length)
        self.predicate_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.functor_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.leaf_net = torch.nn.Linear(self._leaf_factor * self.length + self._leaf_offset, self.length)

        self.embeddings = dict(
            constants=torch.nn.Embedding(self.max_embeddings, self.length).to(self.device),
            functors = torch.nn.Embedding(self.max_embeddings, self.length).to(self.device),
            predicates = torch.nn.Embedding(self.max_embeddings, self.length).to(self.device),
            binary_operators = torch.nn.Embedding(self.max_embeddings, self.length).to(self.device),
            variables = torch.nn.Embedding(self.max_embeddings, self.length).to(self.device))

        self._null = torch.autograd.Variable(torch.randn(self.length)).to(self.device)
        self._masked = torch.zeros(self._leaf_factor * self.length + self._leaf_offset).to(self.device)

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
            return self.embeddings[elem.kind](torch.tensor([elem.value]).to(self.device)).squeeze(0)
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
        return torch.relu(self.binary_formula(torch.cat([self.visit(formula.left),self.visit(formula.operator), self.visit(formula.right)], dim=-1)))

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        arguments = [self.visit(expression.functor)]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(self._null)
        return torch.relu(self.functor_formula(torch.cat(arguments, dim=-1)))

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        arguments = [self.visit(expression.predicate)]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(self._null)
        return torch.relu(self.predicate_formula(torch.cat(arguments, dim=-1)))

    def visit_masked(self, masked: MaskedElement):
        return self._masked


class Residual(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

class PremiseSelector(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.formula_net = FormulaNet()
        self.formula_net.to(self.formula_net.device)
        self.encoder = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.conjecture_squash = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.scoring = torch.nn.Linear(self.formula_net.output_size*3, self.formula_net.output_size)
        self.softmax = torch.nn.Softmax(-1)
        self.decoder = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.final = torch.nn.Linear(self.formula_net.output_size, 1)
        self.device = self.formula_net.device

    def attention_loop(self, premise_stack, hidden, conj, thought):
        ratings = self.softmax(self.scoring(torch.cat((premise_stack, conj, hidden), dim=-1)))
        context = torch.sum(torch.mul(premise_stack, ratings), dim=0, keepdim=True)
        thought = torch.cat((thought, context))
        o, hidden = self.decoder.forward(thought)
        return torch.sigmoid(self.final(o[-1])).squeeze(-1), hidden

    def forward(self, premises, conjectures):
        self.formula_net.prepare(premises, conjectures)
        premise_stack = torch.stack([self.formula_net.forward(p) for p in premises]).unsqueeze(1)
        num_prem = premise_stack.size()[0]
        hidden = premise_stack[-1]
        conj = torch.relu(self.conjecture_squash.forward((torch.stack([self.formula_net.forward(c) for c in conjectures]).unsqueeze(1)))[0].expand(num_prem,1,self.formula_net.output_size))
        thought = torch.empty(1, 1, self.formula_net.output_size).to(self.device)
        output = []
        for i in range(num_prem):
            hidden = hidden.expand(num_prem, 1, self.formula_net.output_size)
            o, hidden = self.attention_loop(premise_stack, hidden, conj, thought)
            output.append(o)
        return torch.cat(output).to(self.device)


class PremiseSelectorGRU(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.formula_net = FormulaNet()
        self.formula_net.to(self.formula_net.device)
        self.device = self.formula_net.device
        self.conjecture_squash = torch.nn.GRU(self.formula_net.output_size, self.formula_net.output_size)
        self.gru = torch.nn.GRU(self.formula_net.output_size*2, self.formula_net.output_size, bidirectional=True)
        self.final = torch.nn.Linear(self.formula_net.output_size * 2, 1)

    def forward(self, data):
        premise_stack = torch.nn.utils.rnn.pad_sequence([torch.stack([self.formula_net.forward(p) for p in batch[0]]) for batch in data])
        conjecture_stack = torch.nn.utils.rnn.pad_sequence(
            [torch.stack([self.formula_net.forward(c) for c in batch[1]]) for batch in
             data])
        num_prem = premise_stack.size()[0]
        batches = premise_stack.size()[1]
        print(num_prem)
        conj = torch.relu(self.conjecture_squash.forward(conjecture_stack)[0].expand(num_prem,batches,self.formula_net.output_size))
        o, h = self.gru(torch.cat((premise_stack, conj), dim=-1))
        return self.final(o).squeeze(-1)


def train_selection(gen):
    net = PremiseSelectorGRU()
    net.to(net.device)
    optimizer = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    learning_curve = []
    for epoch in range(10):
        print("Epoch", epoch)
        i = 0
        batch_loss = 0
        batchnumber = 0
        for data, labels in gen():
            batchnumber += 1
            print("Process batch", batchnumber)
            optimizer.zero_grad()
            predictions = net.forward(data)
            ls = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels]).to(net.device)
            l = loss(predictions, ls)
            l.backward()
            batch_loss += l.item()
            optimizer.step()
            i += len(data)
            print(f"Step {i}: loss {l.item()}")
        learning_curve.append(batch_loss / batchnumber)
        torch.save(net.state_dict(), "mask_encoder.{i}.state".format(i=epoch))
    plt.plot(np.array(learning_curve), 'r')
    plt.savefig("curve.png")
    with open("curve.json", "w") as o:
        o.write(str(learning_curve))
    print('Finished Training')


def train_masked(gen):
    net = FormulaNet()
    net.to(net.device)
    mc = MaskCompiler()
    optimizer = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    learning_curve = []
    for epoch in range(100):
        print("Epoch", epoch)
        i = 0
        batch_loss = 0
        batchnumber = 0
        for batch in gen():
            labels = []
            predictions = []
            optimizer.zero_grad()
            for f in batch:
                net.prepare(f)
                x = mc.visit(f)
                if len(x) != 2:
                    print(f"Could not process {f}. Compiler returned {x}")
                    continue
                formula, lab = x
                label = None
                if isinstance(lab, fol.Constant):
                    label = net._constant_cache[lab.symbol]
                elif isinstance(lab, fol.DefinedConstant):
                    label = net._constant_cache[lab.symbol]
                elif isinstance(lab, fol.DistinctObject):
                    label = net._constant_cache[lab.symbol]
                elif isinstance(lab, tuple):
                    if lab[1] == "predicate":
                        label = net._predicate_cache[lab[0]]
                    elif lab[1] == "functor":
                        label = net._functor_cache[lab[0]]
                if label is None:
                    raise Exception(f"Missing handler for {lab} in {formula}")
                else:
                    labels.append(label)
                    predictions.append(net.forward(formula))
            l = loss(torch.stack(predictions), torch.stack(labels))
            l.backward()
            batch_loss += l.item()
            optimizer.step()
            i += len(batch)
            batchnumber += 1
            print(f"Step {i}: loss {l.item()}")
        learning_curve.append(batch_loss/batchnumber)
    plt.plot(np.array(learning_curve), 'r')
    plt.savefig("curve.png")
    with open("curve.json", "w") as o:
        o.write(str(learning_curve))

    torch.save(net.state_dict(), "mask_encoder.state")
    print('Finished Training')
