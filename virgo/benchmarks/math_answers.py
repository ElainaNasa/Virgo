"""有界精确实数表达式比较；不使用 eval、动态导入或近似容差。"""
import re
from fractions import Fraction
from math import gcd


class UnsupportedExpression(ValueError):
    pass


def clean(value):
    value = {r: c for r, c in value.items() if c}
    if len(value) > 32 or any(c.numerator.bit_length() > 4096 or c.denominator.bit_length() > 4096 for c in value.values()):
        raise UnsupportedExpression('表达式超过资源上限')
    return value


def add(a, b):
    result = dict(a)
    for r, c in b.items():
        result[r] = result.get(r, Fraction(0)) + c
    return clean(result)


def multiply(a, b):
    result = {}
    for r, c in a.items():
        for s, d in b.items():
            g = gcd(r, s)
            radical = (r // g) * (s // g)
            if radical > 10**12:
                raise UnsupportedExpression('根式过大')
            result[radical] = result.get(radical, Fraction(0)) + c * d * g
    return clean(result)


def divide(a, b):
    if not b:
        raise UnsupportedExpression('除零')
    if len(b) == 1:
        r, c = next(iter(b.items()))
        return multiply(a, {r: 1 / (c * r)})
    if len(b) == 2:
        (r, c), (s, d) = b.items()
        denominator = c*c*r - d*d*s
        if denominator:
            return divide(multiply(a, {r:c, s:-d}), {1:denominator})
    raise UnsupportedExpression('不支持该分母形式')


def radical(value):
    if not value:
        return {}
    if set(value) != {1} or value[1] < 0:
        raise UnsupportedExpression('仅支持非负有理数的平方根')
    x = value[1]
    n = x.numerator * x.denominator
    if n > 1_000_000:
        raise UnsupportedExpression('根号内数值过大')
    outside, inside, factor = 1, 1, 2
    while factor * factor <= n:
        exponent = 0
        while n % factor == 0:
            n //= factor
            exponent += 1
        outside *= factor ** (exponent // 2)
        if exponent % 2:
            inside *= factor
        factor += 1
    inside *= n
    return {inside: Fraction(outside, x.denominator)}


class Parser:
    def __init__(self, text):
        if len(text) > 256:
            raise UnsupportedExpression('输入过长')
        text = text.replace('−', '-').replace(r'\cdot', '*').replace(r'\times', '*')
        pattern = r'\d+(?:\.\d+)?|\\sqrt|\\frac|[+*/^(){}-]'
        self.tokens = re.findall(pattern, text)
        if ''.join(self.tokens) != text or len(self.tokens) > 128:
            raise UnsupportedExpression('含不支持的符号')
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self):
        value = self.peek()
        if value is None:
            raise UnsupportedExpression('表达式不完整')
        self.index += 1
        return value

    def expression(self):
        value = self.term()
        while self.peek() in ('+', '-'):
            op = self.take()
            other = self.term()
            value = add(value, other if op == '+' else {r:-c for r,c in other.items()})
        return value

    def term(self):
        value = self.unary()
        while True:
            token = self.peek()
            if token in ('*', '/'):
                self.take()
                other = self.unary()
                value = multiply(value, other) if token == '*' else divide(value, other)
            elif token in ('(', '{', r'\sqrt', r'\frac'):
                value = multiply(value, self.unary())
            else:
                break
        return value

    def unary(self):
        if self.peek() in ('+', '-'):
            op = self.take()
            value = self.unary()
            return value if op == '+' else {r:-c for r,c in value.items()}
        value = self.atom()
        if self.peek() == '^':
            self.take()
            power = self.unary()
            if not power:
                exponent = 0
            elif set(power) == {1} and power[1].denominator == 1:
                exponent = int(power[1])
            else:
                raise UnsupportedExpression('仅支持整数幂')
            if abs(exponent) > 12:
                raise UnsupportedExpression('幂次过大')
            result = {1:Fraction(1)}
            for _ in range(abs(exponent)):
                result = multiply(result, value)
            value = divide({1:Fraction(1)}, result) if exponent < 0 else result
        return value

    def atom(self):
        token = self.take()
        if token in ('(', '{'):
            closing = ')' if token == '(' else '}'
            value = self.expression()
            if self.take() != closing:
                raise UnsupportedExpression('括号不匹配')
            return value
        if token == r'\sqrt':
            return radical(self.atom())
        if token == r'\frac':
            numerator = self.atom()
            return divide(numerator, self.atom())
        if re.fullmatch(r'\d+(?:\.\d+)?', token):
            return clean({1:Fraction(token)})
        raise UnsupportedExpression('不支持的表达式')

    def parse(self):
        value = self.expression()
        if self.peek() is not None:
            raise UnsupportedExpression('有未消费的内容')
        return value


def exact_expression_equal(left, right):
    """只判定支持的精确常数表达式；不支持或超限时保守返回 False。"""
    if left is None or right is None:
        return False
    try:
        return Parser(left).parse() == Parser(right).parse()
    except (UnsupportedExpression, ArithmeticError, RecursionError, ValueError):
        return False
