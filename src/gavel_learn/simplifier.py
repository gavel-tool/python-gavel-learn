from gavel.dialects.base.compiler import Compiler
from gavel.logic import logic as fol, problem
import random


class EncodedElement(fol.LogicElement):
    __visit_name__ = "encoded_element"

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value


class MapExtractor(Compiler):

    def __init__(self):
        self.maps = dict(
            predicates=dict(),
            constants=dict(),
            functors=dict(),
            binary_operators=dict(),
            variables=dict()
        )

    def _set(self, key, element, **kwargs):
        d = self.maps[key]
        try:
            return d[element]
        except KeyError:
            e = EncodedElement(key, len(d)+1)
            d[element] = e
            return e

    def visit_defined_predicate(self, predicate: fol.DefinedPredicate, **kwargs):
        return self._set("predicates", predicate.value, **kwargs)

    def visit_unary_formula(self, formula: fol.UnaryFormula, **kwargs):
        return fol.UnaryFormula(connective=formula.connective, formula=self.visit(formula.formula, **kwargs))

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula, **kwargs):
        return fol.QuantifiedFormula(
            quantifier=formula.quantifier, variables=[self.visit(v) for v in formula.variables], formula=self.visit(formula.formula, **kwargs))

    def visit_annotated_formula(self, anno: problem.AnnotatedFormula, **kwargs):
        return self.visit(anno.formula, **kwargs)

    def visit_binary_formula(self, formula: fol.BinaryFormula, **kwargs):
        return fol.BinaryFormula(
            left=self.visit(formula.left, **kwargs),
            operator=self._set("binary_operators", formula.operator, **kwargs),
            right=self.visit(formula.right, **kwargs))

    def visit_functor_expression(self, expression: fol.FunctorExpression, **kwargs):
        return fol.FunctorExpression(
            functor=self._set("functors", expression.functor, **kwargs),
            arguments=[self.visit(argument, **kwargs) for argument in expression.arguments])

    def visit_predicate_expression(self, expression: fol.PredicateExpression, **kwargs):
        return fol.PredicateExpression(
            predicate=self._set("predicates", expression.predicate, **kwargs),
            arguments=[self.visit(argument, **kwargs) for argument in expression.arguments])

    def visit_variable(self, variable: fol.Variable, **kwargs):
        return self._set("variables", variable.symbol, **kwargs)

    def visit_constant(self, constant: fol.Constant, **kwargs):
        return self._set("constants", constant.symbol, **kwargs)

    def visit_distinct_object(self, obj: fol.DistinctObject, **kwargs):
        return self._set("constants", obj.symbol, **kwargs)

    def visit_defined_constant(self, obj: fol.DefinedConstant, **kwargs):
        return self._set("constants", obj.value, **kwargs)

    def visit_binary_connective(self, connective: fol.BinaryConnective, **kwargs):
        return self._set("binary_connective", connective, **kwargs)

    def visit_predefined_constant(self, obj: fol.PredefinedConstant, **kwargs):
        return self._set("constants", obj.value, **kwargs)

    def visit_problem(self, p: problem.Problem, **kwargs):
        return problem.Problem(
            premises=[self.visit(line, **kwargs) for line in p.premises],
            conjectures=[self.visit(line, **kwargs) for line in p.conjectures],
            imports=p.imports
        )


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

    def visit_distinct_object(self, obj: fol.DistinctObject):
        return obj, obj

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
