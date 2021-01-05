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
from gavel_db.dialects.db.parser import DBLogicParser
from gavel_db.dialects.db.structures import Formula
from gavel_learn.learn import train_masked
import pickle
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
            for line in f.readlines():
                yield p._parse_rec(line)

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


def _batchify(g, batch_size):
    def inner():
        batch = []
        for x in g():
            batch.append(x)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    return inner


def learn_memory(gen, m, b):
    g = _batchify(gen, b)
    if m:
        _cache = list(g())

        def g2():
            return _cache
        return train_masked(g2)
    else:
        train_masked(g)


learn.add_command(learn_masked)
learn.add_command(learn_masked_db)
