"""Single-edit functional mutants, split by operator family."""
from __future__ import annotations

import ast
import copy
import random


def mutations(code: str, pool: str, seed: int = 0, limit: int = 24,
              sequence_parameters: set[str] | None = None) -> list[dict]:
    tree = ast.parse(code)
    changes = []
    for index, node in enumerate(ast.walk(tree)):
        if pool == 'selection':
            if isinstance(node, ast.BinOp):
                operators = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]
                if type(node.op) in operators + [ast.Div, ast.Pow, ast.BitXor, ast.BitAnd, ast.BitOr]:
                    for operator in operators:
                        if type(node.op) is not operator:
                            changes.append((index, 'op', operator(), 'arithmetic'))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                swap = {'min':['max','sum','len'], 'max':['min','sum','len'],
                        'all':['any'], 'any':['all'], 'sum':['max','min','len'],
                        'sorted':['list','reversed'], 'len':['sum','min','max']}
                if node.func.id in swap:
                    for other in swap[node.func.id]:
                        changes.append((index, 'func', ast.Name(id=other, ctx=ast.Load()), 'aggregation'))
                if node.func.id == 'range' and node.args:
                    for offset in [-1, 1]:
                        args = copy.deepcopy(node.args)
                        position = 0 if len(args) == 1 else 1
                        args[position] = ast.BinOp(left=args[position], op=ast.Add(), right=ast.Constant(offset))
                        changes.append((index, 'args', args, 'loop_bound'))
            if isinstance(node, ast.BoolOp):
                changes.append((index, 'op', ast.Or() if isinstance(node.op, ast.And) else ast.And(), 'logical'))
            if isinstance(node, ast.Call):
                for i, keyword in enumerate(node.keywords):
                    if keyword.arg in {'key', 'reverse'}:
                        changes.append((index, 'keywords', copy.deepcopy(node.keywords[:i] + node.keywords[i+1:]), 'ordering_keyword'))
                if isinstance(node.func, ast.Name) and node.func.id in {'sorted', 'min', 'max'}:
                    for i, keyword in enumerate(node.keywords):
                        if keyword.arg == 'key':
                            for expression in ['lambda x: x', 'lambda x: -len(x)']:
                                keywords = copy.deepcopy(node.keywords)
                                keywords[i].value = ast.parse(expression, mode='eval').body
                                changes.append((index, 'keywords', keywords, 'ordering_keyword'))
            if isinstance(node, ast.comprehension) and node.ifs:
                for i in range(len(node.ifs)):
                    changes.append((index, 'ifs', copy.deepcopy(node.ifs[:i] + node.ifs[i+1:]), 'filter_omission'))
            if isinstance(node, ast.If) and node.orelse:
                changes.append((index, 'body', copy.deepcopy(node.orelse), 'branch_replacement'))
                changes.append((index, 'orelse', copy.deepcopy(node.body), 'branch_replacement'))
            if isinstance(node, ast.Attribute) and node.attr in {'startswith','endswith','lower','upper','lstrip','rstrip'}:
                swap = {'startswith':'endswith','endswith':'startswith','lower':'upper','upper':'lower','lstrip':'rstrip','rstrip':'lstrip'}
                changes.append((index, 'attr', swap[node.attr], 'string_operation'))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                changes.append((index, 'op', ast.UAdd(), 'sign'))
        else:
            if isinstance(node, ast.Compare):
                alternatives = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
                for i, op in enumerate(node.ops):
                    for other in alternatives if type(op) in alternatives else []:
                        if type(op) is other: continue
                        ops = copy.deepcopy(node.ops); ops[i] = other()
                        changes.append((index, 'ops', ops, 'comparison'))
                    if isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)):
                        ops = copy.deepcopy(node.ops)
                        ops[i] = {ast.In:ast.NotIn, ast.NotIn:ast.In, ast.Is:ast.IsNot, ast.IsNot:ast.Is}[type(op)]()
                        changes.append((index, 'ops', ops, 'comparison'))
            if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 100000:
                for shift in [-2, -1, 1, 2]: changes.append((index, 'value', node.value + shift, 'constant_boundary'))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in (sequence_parameters or set()):
                    for section in ['1:', ':-1', '::-1']:
                        replacement = ast.parse(f'{node.id}[{section}]', mode='eval').body
                        changes.append((index, '__replace__', replacement, 'input_slice'))
                for function in tree.body:
                    if not isinstance(function, ast.FunctionDef): continue
                    parameters = [a.arg for a in function.args.args]
                    if node.id not in parameters or node not in list(ast.walk(function)): continue
                    for other in parameters:
                        if other != node.id: changes.append((index, 'id', other, 'argument_substitution'))
            if isinstance(node, ast.Compare):
                for operand in [node.left] + node.comparators:
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, str) and operand.value:
                        target_index = next(i for i,n in enumerate(ast.walk(tree)) if n is operand)
                        for value in {operand.value[1:], operand.value[:-1], operand.value.swapcase()} - {operand.value}:
                            changes.append((target_index, 'value', value, 'string_boundary'))
    random.Random(seed).shuffle(changes)
    output, seen = [], {ast.unparse(tree)}
    for index, field, value, operator in changes:
        mutant = copy.deepcopy(tree)
        target = list(ast.walk(mutant))[index]
        if field == '__replace__':
            class Replace(ast.NodeTransformer):
                def visit(self, node):
                    return copy.deepcopy(value) if node is target else super().visit(node)
            mutant = Replace().visit(mutant)
        else:
            setattr(target, field, value)
        try:
            source = ast.unparse(ast.fix_missing_locations(mutant))
            compile(source, '<mutant>', 'exec')
        except (SyntaxError, ValueError): continue
        if source in seen: continue
        seen.add(source)
        output.append({'code':source, 'source':'reference_mutation', 'operator':operator, 'pool':pool})
        if len(output) >= limit: break
    return output
