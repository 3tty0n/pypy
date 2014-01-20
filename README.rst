==============================================
Unipycation: A Language Composition Experiment
==============================================

Unipycation is an experimental composition of a Python interpreter (PyPy
http://pypy.org/) and a Prolog interpreter (Pyrolog by Carl Friedrich
Bolz https://bitbucket.org/cfbolz/pyrolog). The languages are composed
using RPython, meaning that the entire composition is meta-traced.

The goal of the project is to identify the challenges associated with composing 
programming languages whose paradigms differ vastly and to evaluate RPython as
a language composition platform.

Setup
=====

Run the setup script:

    $ python2.7 bootstrap.py

Follow instructions printed to stdout.

On a 64-bit architecture the translation process will consume about 8GB of
memory at peak. The resulting pypy-c binary is the composed Python/Prolog
compiler.

Using Unipycation
=================

For the moment, the languages are composed without any adjustments to
syntax. In other words, communication between Python and Prolog is in
the form of an API. Better syntactic composition will come later.

The interface is described in the paper `Unipycation: A Case Study in
Cross-Language Tracing
<http://soft-dev.org/pubs/pdf/barrett_bolz_tratt__unipycation_a_study_in_cross_language_tracing.pdf>`_
which appeared in VMIL'13.

The interface is subject to change.

Authors
=======

Unipycation is authored by Edd Barrett and Carl Friedrich Bolz.
