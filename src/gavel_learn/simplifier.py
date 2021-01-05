from gavel.dialects.base.compiler import Compiler
from gavel.logic import logic as fol, problem
import random

class MapExtractor(Compiler):

    def visit_quantifier(self, quantifier: fol.Quantifier):
        pass

    def visit_defined_predicate(self, predicate: fol.DefinedPredicate, **kwargs):
        kwargs["predicates"].add(predicate.value)

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        self.visit(formula.formula, **kwargs)

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        self.visit(formula.formula, **kwargs)

    def visit_annotated_formula(self, anno: problem.AnnotatedFormula, **kwargs):
        return self.visit(anno.formula, **kwargs)

    def visit_binary_formula(self, formula: fol.BinaryFormula, **kwargs):
        self.visit(formula.left, **kwargs),
        kwargs["binary_operators"].add(formula.operator),
        self.visit(formula.right, **kwargs)

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        kwargs["functors"].add(expression.functor)
        for argument in expression.arguments:
            self.visit(argument, **kwargs)

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        kwargs["predicates"].add(expression.predicate)
        for argument in expression.arguments:
            self.visit(argument, **kwargs)

    def visit_variable(self, variable: fol.Variable, **kwargs):
        kwargs["variables"].add(variable.symbol)

    def visit_constant(self, constant: fol.Constant, **kwargs):
        kwargs["constants"].add(constant.symbol)

    def visit_distinct_object(self, obj: fol.DistinctObject, **kwargs):
        kwargs["constants"].add(obj.symbol)

    def visit_defined_constant(self, obj: fol.DefinedConstant, **kwargs):
        kwargs["constants"].add(obj.value)

    def visit_problem(self, problem: problem.Problem, **kwargs):
        for line in problem.premises:
            self.visit(line)


class MaskedElement(fol.LogicElement):
    __visit_name__ = "masked"

    def __init__(self, original):
        self.original = original


class MaskCompiler(Compiler):

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        obj, masked = self.visit(formula.formula, **kwargs)
        return fol.UnaryFormula(connective=formula.connective, formula=obj), masked

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        form, masked = self.visit(formula.formula, **kwargs)
        return fol.QuantifiedFormula(formula.quantifier, formula.variables, form), masked

    def visit_binary_formula(self, formula: fol.BinaryFormula, **kwargs):
        masked = None
        l = formula.left
        o = formula.operator
        r = formula.right
        ran = 2*random.random()
        if ran <= 1:
            l, masked = self.visit(formula.left, **kwargs)
        if not masked:
            r, masked = self.visit(formula.right, **kwargs)
        if not masked:
            l, masked = self.visit(formula.left, **kwargs)
        return fol.BinaryFormula(left=l, operator=o, right=r), masked

    def visit_constant(self, constant: fol.Constant):
        return constant, constant

    def visit_defined_constant(self, constant: fol.DefinedConstant):
        return constant, constant

    def visit_variable(self, variable: fol.Variable):
        return variable, None

    def forp(self, name, arguments, t, **kwargs):
        masked = None
        s = list(range(len(arguments)+1))
        random.shuffle(s)
        while s and masked is None:
            i = s.pop()
            a = []
            if i == 0:
                masked = (name, t)
            for j, argument in enumerate(arguments, 1):
                if j == i:
                    argument, masked = self.visit(argument, **kwargs)
                a.append(argument)
        return a, masked

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        a, masked = self.forp(expression.functor,expression.arguments, "functor", **kwargs)
        return fol.FunctorExpression(functor=expression.functor, arguments=a), masked

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        a, masked = self.forp(expression.predicate,expression.arguments, "predicate", **kwargs)
        return fol.PredicateExpression(predicate=expression.predicate, arguments=a), masked
