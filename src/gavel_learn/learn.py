import torch
from gavel.logic import logic as fol
from gavel.logic import problem
from gavel.dialects.base.compiler import Compiler
from gavel_learn.simplifier import MapExtractor


class FormulaNet(torch.nn.Module, Compiler):

    def __init__(self):
        super().__init__()
        self.length = 100
        self.argument_limit = 5
        self.unary_formula = torch.nn.Linear(self.length, self.length)
        self.binary_formula = torch.nn.Linear(3*self.length, self.length)
        self.predicate_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self.functor_formula = torch.nn.Linear((1 + self.argument_limit) * self.length, self.length)
        self._constant_cache = None
        self._functor_cache = None
        self._predicate_cache = None
        self._binary_operator_cache = None
        self._variables_cache = None

    def prepare(self, p: problem.Problem):
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
        self._binary_operator_cache = dict(map(self.encode, enumerate(maps["binary_operators"])))
        self._variables_cache = dict(map(self.encode, enumerate(maps["variables"])))

    def encode(self, i, o):
        v = torch.zeros(self.length)
        v[i] = 1
        return o, v

    def visit_defined_predicate(self, predicate: fol.DefinedPredicate, **kwargs):
        return self._predicate_cache(predicate.value)

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        return self.unary_formula(self.visit(formula.formula, **kwargs))

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        pass

    def visit_annotated_formula(self, anno: problem.AnnotatedFormula, **kwargs):
        return self.visit(anno.formula)

    def visit_binary_formula(self, formula: fol.BinaryFormula):
        return self.binary_formula(self.visit(formula.left),self.visit(formula.operator), self.visit(formula.right))

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        arguments = [self.visit(a, **kwargs) for a in expression.arguments]
        # Fill missing arguments with zeros
        for _ in range(len(arguments), self.argument_limit):
            arguments.append(torch.zeros(self.length))
        arguments = [self.visit(expression.functor, **kwargs)] + arguments
        return self.functor_formula(torch.cat(arguments))

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        arguments = [self.visit(a, **kwargs) for a in expression.arguments]
        # Fill missing arguments with zeros
        for _ in range(len(arguments), self.argument_limit):
            arguments.append(torch.zeros(self.length))
        arguments = [self.visit(expression.predicate, **kwargs)] + arguments
        return self.functor_formula(torch.cat(arguments))

    def visit_variable(self, variable: fol.Variable, **kwargs):
        return kwargs["variables"][variable.symbol]

    def visit_constant(self, variable: fol.Variable, **kwargs):
        return kwargs["variables"][variable.symbol]

    def visit_distinct_object(self, obj: fol.DistinctObject, **kwargs):
        return kwargs["constants"][obj.symbol]

    def visit_defined_constant(self, obj: fol.DefinedConstant, **kwargs):
        return kwargs["constants"][obj.value]


class PremiseSelector(torch.nn.Module):

    def __init__(self):
        super().__init__()
