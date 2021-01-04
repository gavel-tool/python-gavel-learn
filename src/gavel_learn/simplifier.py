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

    masked_entities = {b: a for a, b in enumerate([
        fol.Quantifier.EXISTENTIAL,
        fol.Quantifier.UNIVERSAL,
        fol.BinaryConnective.APPLY,
        fol.BinaryConnective.ARROW,
        fol.BinaryConnective.ASSIGN,
        fol.BinaryConnective.BIIMPLICATION,
        fol.BinaryConnective.CONJUNCTION,
        fol.BinaryConnective.DISJUNCTION,
        fol.BinaryConnective.EQ,
        fol.BinaryConnective.IMPLICATION,
        fol.BinaryConnective.NEGATED_CONJUNCTION,
        fol.BinaryConnective.NEGATED_DISJUNCTION,
        fol.BinaryConnective.NEQ,
        fol.BinaryConnective.PRODUCT,
        fol.BinaryConnective.REVERSE_IMPLICATION,
        fol.BinaryConnective.SIMILARITY,
        fol.BinaryConnective.UNION])
    }

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        obj, masked = self.visit(formula.formula, **kwargs)
        return fol.UnaryFormula(connective=formula.connective, formula=obj), masked

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        if random.random() <= 0.5:
            quantifier, masked = self.visit(formula.quantifier, **kwargs)
            return fol.QuantifiedFormula(quantifier, formula.variables, formula.formula), masked
        else:
            form, masked = self.visit(formula.formula, **kwargs)
            return fol.QuantifiedFormula(formula.quantifier, formula.variables, form), masked

    def visit_binary_formula(self, formula: fol.BinaryFormula, **kwargs):
        ran = 3*random.random()

        l = formula.left
        o = formula.operator
        r = formula.right
        masked = None
        if ran <= 1:
            l, masked = self.visit(formula.left, **kwargs)
        if not masked:
            r, masked = self.visit(formula.right, **kwargs)
        if not masked:
            o, masked = self.visit(formula.operator, **kwargs)
        return fol.BinaryFormula(left=l, operator=o, right=r), masked

    def visit_constant(self, constant: fol.Constant):
        return constant, None

    def visit_variable(self, variable: fol.Variable):
        return variable, None

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        i = random.random()*len(expression.arguments)
        a = []
        masked = None
        for j, argument in enumerate(expression.arguments):
            if j == i:
                argument, masked = self.visit(argument, **kwargs)
            a.append(argument)
        return fol.FunctorExpression(functor=expression.functor, arguments=a), masked

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        i = random.random() * len(expression.arguments)
        a = []
        masked = None
        for j, argument in enumerate(expression.arguments):
            if j == i:
                argument, masked = self.visit(argument, **kwargs)
            a.append(argument)
        return fol.PredicateExpression(predicate=expression.predicate, arguments=a), masked

    def visit_quantifier(self, quantifier: fol.Quantifier):
        return MaskedElement(quantifier), self.masked_entities[quantifier]

    def visit_binary_connective(self, connective: fol.BinaryConnective):
        return MaskedElement(connective), self.masked_entities[connective]
