import torch
from gavel.logic import logic as fol
from gavel.logic import problem
from gavel.dialects.base.compiler import Compiler
from gavel_learn.simplifier import MapExtractor, MaskCompiler, MaskedElement


class FormulaNet(torch.nn.Module, Compiler):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self._leaf_factor = 5
        self._leaf_offset = 5
        self.length = 50
        self.argument_limit = 5
        self.unary_formula = torch.nn.Linear(self.length, self.length)
        self.final = torch.nn.Linear(self.length, self._leaf_factor*self.length + self._leaf_offset)
        self.existential_quant = torch.nn.Linear(self.length, self.length)
        self.universal_quant = torch.nn.Linear(self.length, self.length)
        self.binary_formula = torch.nn.Linear(3*self.length, self.length)
        self.predicate_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.functor_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.leaf_net = torch.nn.Linear(self._leaf_factor * self.length  + self._leaf_offset, self.length)
        self._constant_cache = None
        self._functor_cache = None
        self._predicate_cache = None
        self._binary_operator_cache = None
        self._variables_cache = None

    def forward(self, input: fol.LogicExpression):
        return self.final(self.visit(input))

    def prepare(self, p):
        maps = dict(
            predicates=set(),
            constants=set(),
            functors=set(),
            binary_operators=set(),
            variables=set()
        )
        MapExtractor().visit(p,**maps)
        self._constant_cache = dict(map(self.encode(0), enumerate(maps["constants"])))
        self._functor_cache = dict(map(self.encode(1), enumerate(maps["functors"])))
        self._predicate_cache = dict(map(self.encode(2), enumerate(maps["predicates"])))
        self._binary_operator_cache = dict(map(self.encode(3), enumerate(fol.BinaryConnective)))
        self._variables_cache = dict(map(self.encode(4), enumerate(maps["variables"])))

    def encode(self, x):
        def inner(y):
            i, o = y
            v = torch.zeros(self._leaf_factor * self.length + self._leaf_offset).to(self.device)
            if i < self._leaf_factor * self.length:
                v[i + self._leaf_offset] = 1
            v[x] = 1
            return o, v
        return inner

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
            arguments.append(torch.zeros(self.length).to(self.device))
        return torch.relu(self.functor_formula(torch.cat(arguments, dim=-1)))

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        arguments = [self.leaf_net(self._predicate_cache[expression.predicate])]
        arguments += [self.visit(a, **kwargs) for a in expression.arguments[:min(len(expression.arguments), self.argument_limit)]]
        # Fill missing arguments with zeros
        for _ in range(len(arguments)+1, self.argument_limit+2):
            arguments.append(torch.zeros(self.length).to(self.device))
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
        return torch.zeros(self.length).to(self.device)


class PremiseSelector(torch.nn.Module):

    def __init__(self):
        super().__init__()


def train_masked(gen):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    net = FormulaNet(device=device)
    net.to(device)
    mc = MaskCompiler()
    optimizer = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    running_loss = 0.0
    for epoch in range(10):
        print("Epoch", epoch)

        for batch in gen():
            labels = []
            predictions = []
            optimizer.zero_grad()
            for f in batch:
                net.prepare(f)
                formula, lab = mc.visit(f)
                label = None
                if isinstance(lab, fol.Constant):
                    label = net._constant_cache[lab.symbol]
                elif isinstance(lab, fol.DefinedConstant):
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
            optimizer.step()
            print(f"loss {l.item()}")
    torch.save(net.state_dict(), "mask_encoder.state")
    print('Finished Training')
