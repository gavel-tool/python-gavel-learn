from gavel.dialects.base.compiler import Compiler
from gavel.logic import logic as fol, problem


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
        kwargs["functors"].add(expression.predicate)
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
