import torch
from gavel.logic import logic as fol
from gavel.logic import problem
from gavel.dialects.base.compiler import Compiler
from gavel_learn.simplifier import MapExtractor, MaskCompiler, MaskedElement
from gavel_db.dialects.db.connection import with_session
from gavel_db.dialects.db.structures import Formula
from gavel_db.dialects.db.parser import DBLogicParser
from gavel_db.dialects.db.compiler import JSONCompiler
import pandas as pd

class FormulaNet(torch.nn.Module, Compiler):

    def __init__(self):
        super().__init__()
        self._leaf_factor = 5
        self.length = 50
        self.argument_limit = 5
        self.unary_formula = torch.nn.Linear(self.length, self.length)
        self.existential_quant = torch.nn.Linear(self.length, self.length)
        self.universal_quant = torch.nn.Linear(self.length, self.length)
        self.binary_formula = torch.nn.Linear(3*self.length, self.length)
        self.predicate_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.functor_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.leaf_net = torch.nn.Linear(self._leaf_factor * self.length, self.length)
        self._constant_cache = None
        self._functor_cache = None
        self._predicate_cache = None
        self._binary_operator_cache = None
        self._variables_cache = None

    def forward(self, input: fol.LogicExpression):
        return self.visit(input)

    def prepare(self, p):
        maps = dict(
            predicates=set(),
            constants=set(),
            functors=set(),
            binary_operators=set(),
            variables=set()
        )
        MapExtractor().visit(p,**maps)
        self._constant_cache = dict(map(self.encode, enumerate(maps["constants"])))
        self._functor_cache = dict(map(self.encode, enumerate(maps["functors"])))
        self._predicate_cache = dict(map(self.encode, enumerate(maps["predicates"])))
        self._binary_operator_cache = dict(map(self.encode, enumerate(fol.BinaryConnective)))
        self._variables_cache = dict(map(self.encode, enumerate(maps["variables"])))

    def encode(self, x):
        i, o = x
        v = torch.zeros(self._leaf_factor * self.length)
        if i < self._leaf_factor * self.length:
            v[i] = 1
        else:
            print(x)
        return o, v

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
        arguments = [self.leaf_net(self._functor_cache[expression.functor])]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(torch.zeros(self.length))
        return torch.relu(self.functor_formula(torch.cat(arguments, dim=-1)))

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        arguments = [self.leaf_net(self._predicate_cache[expression.predicate])]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(torch.zeros(self.length))
        return torch.relu(self.predicate_formula(torch.cat(arguments, dim=-1)))

    def visit_variable(self, variable: fol.Variable, **kwargs):
        return torch.relu(self.leaf_net(self._variables_cache[variable.symbol]))

    def visit_constant(self, obj: fol.Constant, **kwargs):
        return torch.relu(self.leaf_net(self._constant_cache[obj.symbol]))

    def visit_distinct_object(self, obj: fol.DistinctObject, **kwargs):
        return torch.relu(self.leaf_net(self._constant_cache[obj.symbol]))

    def visit_defined_constant(self, obj: fol.DefinedConstant, **kwargs):
        return torch.relu(self.leaf_net(self._constant_cache[obj.value]))

    def visit_binary_connective(self, connective: fol.BinaryConnective):
        return torch.relu(self.leaf_net(self._binary_operator_cache[connective]))

    def visit_defined_predicate(self, predicate: fol.DefinedPredicate, **kwargs):
        return torch.relu(self.leaf_net(self._predicate_cache(predicate.value)))

    def visit_masked(self, masked: MaskedElement):
        return torch.zeros(self.length)


class PremiseSelector(torch.nn.Module):

    def __init__(self):
        super().__init__()


def train_masked(gen):
    net = FormulaNet()
    mc = MaskCompiler()
    optimizer = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    running_loss = 0.0
    for epoch in range(10):
        print("Epoch", epoch)
        i = 0
        for f in gen(): #session.query(Formula.json).yield_per(100):
            net.prepare(f)
            optimizer.zero_grad()
            lab = None
            j = 0
            if isinstance(f, str):
                continue
            while lab is None:
                if j > 100:
                    break
                ret = mc.visit(f)
                if len(ret) == 2:
                    formula, lab = ret
                j+=1
            if lab:
                label = torch.zeros(net.length)
                label[lab] = 1
                prediction = net.forward(formula)
                l = loss(prediction, label)
                l.backward()
                optimizer.step()
                running_loss += l.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                i += 1
    torch.save(net.state_dict(), "mask_encoder.state")
    print('Finished Training')
