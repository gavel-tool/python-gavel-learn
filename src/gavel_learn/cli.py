"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mgavel_db` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``gavel_db.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``gavel_db.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
__all__ = ["learn"]
import os

import gavel.config.settings as settings
from gavel_db.dialects.db.connection import with_session
from gavel_db.dialects.db.parser import DBLogicParser, DBProblemParser
from gavel_db.dialects.db.structures import Formula
from gavel_db.dialects.db import structures
from gavel_learn.learn import train_masked, train_selection
from gavel_learn.simplifier import MapExtractor
from gavel.dialects.tptp.parser import TPTPProblemParser, TPTPParser
import json
import click


@click.group()
def learn():
    pass


@click.command()
@click.argument("path", default=None)
@click.argument("batch", default=None, type=int)
@click.option("--m", default=False)
def learn_masked(path, batch, m=False):
    def gen():
        p = DBLogicParser()
        with open(path,"r") as f:
            for line in f:
                yield p._parse_rec(json.loads(line))

    learn_memory(gen,m, batch)

@click.command()
@click.argument("batch", default=None, type=int)
@click.option("--m", default=False)
def learn_masked_db(batch, m=False):
    @with_session
    def gen(session):
        p = DBLogicParser()
        for dbf in session.query(Formula.json).yield_per(10):
            yield p._parse_rec(dbf[0])

    learn_memory(gen, m, batch)


@click.command()
@click.argument("batch", default=None, type=int)
@click.option("--m", default=False)
def learn_selection_db(batch, m=False):
    @with_session
    def gen(session):
        parser = DBLogicParser()
        for solution in session.query(structures.Solution).yield_per(1):
            premises = []
            used = []
            for prem in solution.problem.all_premises(session):
                premises.append(parser._parse_rec(prem.json))
                used.append(1.0 if prem in solution.premises else 0.0)
            conjectures = [parser._parse_rec(c.json) for c in solution.problem.conjectures]
            if premises:
                yield (premises, conjectures), used
    learn_memory(gen, m, batch)


@click.command()
@click.argument("path", default=None)
@click.argument("batch", default=None, type=int)
@click.option("--m", default=False)
def learn_selection(path, batch, m=False):

    def gen():
        lparser= TPTPParser()
        pparser = TPTPProblemParser()
        sources = {}
        with open(path, "r") as f:
            for row in json.load(f):
                me = MapExtractor()
                problem = pparser.parse_from_file(os.path.join(settings.TPTP_ROOT,row["path"]))
                premises = problem.premises
                for imp in problem.imports:
                    try:
                        imp_prem = sources[imp.path]
                    except KeyError:
                        imp_prem = list(lparser.parse_from_file(os.path.join(settings.TPTP_ROOT, imp.path)))
                        sources[imp.path] = imp_prem
                    premises += imp_prem
                used = []
                for prem in premises:
                    used.append(1.0 if prem.name in row["used"] else 0.0)
                mapped_premises = [me.visit(prem) for prem in premises]
                conjectures = [me.visit(c) for c in problem.conjectures]
                if mapped_premises:
                    yield (mapped_premises, conjectures), used
    learn_memory(gen, m, batch)


def _batchify(g, batch_size):
    def inner():
        data_batch = []
        label_batch = []
        for x, y in g():
            data_batch.append(x)
            label_batch.append(y)
            if len(data_batch) >= batch_size:
                yield data_batch, label_batch
                data_batch = []
                label_batch = []
        if data_batch:
            yield data_batch, label_batch
    return inner


def learn_memory(gen, m, b):
    g = _batchify(gen, b)
    if m:
        _cache = list(g())

        def g2():
            return _cache
        return train_selection(g2)
    else:
        return train_selection(g)


learn.add_command(learn_masked)
learn.add_command(learn_masked_db)
learn.add_command(learn_selection)
learn.add_command(learn_selection_db)
