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
from gavel_db.dialects.db.parser import DBLogicParser
from gavel_learn.learn import train_masked
import pickle
import click


@click.group()
def learn():
    pass


@click.command()
@click.argument("path", default=None)
@click.option("--m", default=False)
def learn_masked(path, m=False):
    def gen():
        with open(path,"rb") as f:
            while True:
                yield pickle.load(f)
    if m:
        _cache = list(gen())

        def g2():
            return _cache
        return train_masked(g2)
    else:
        train_masked(gen)
    print("Done:", path)


learn.add_command(learn_masked)
