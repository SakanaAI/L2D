from argparse import Namespace
import re
import string
from typing import Union


def extract_first_choice_answer(
        completion: str, choices: list | int, return_index: bool = True):
    if isinstance(choices, int):
        num_choices = choices
        possible_choices = string.ascii_uppercase[:num_choices]
    else:
        num_choices = len(choices)
        possible_choices = choices

    options = "".join(possible_choices)

    answer_regex = f"([{options}])" + r"(?:[^a-zA-Z0-9]|$)"
    matches = re.findall(answer_regex, completion)
    if not matches:

        if not matches:
            return None, None
    text = matches[0]
    match_idx = re.search(answer_regex, completion).start()
    if return_index:
        return possible_choices.index(text), match_idx
    return text, match_idx


def extract_last_number(completion: str):
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))


def fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except (AssertionError, Exception):
        return string


def remove_right_units(string):

    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def strip_math_latex_string(string):

    string = string.replace('\n', '')

    string = string.replace('\\!', '')

    string = string.replace('\\\\', '\\')

    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    string = string.replace('\\$', '')

    string = remove_right_units(string)

    string = string.replace('\\%', '')
    string = string.replace('\%', '')

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = fix_sqrt(string)

    string = string.replace(' ', '')

    string = fix_fracs(string)

    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()

    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string


def extract_last_boxed_only_string(completion):
    idx = completion.rfind('\\boxed')
    if idx < 0:
        idx = completion.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(completion):
        if completion[i] == '{':
            num_left_braces_open += 1
        if completion[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = completion[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_fraction_components(latex_fraction):
    match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", latex_fraction)
    if not match:
        print(f"Invalid LaTeX fraction: {latex_fraction}, unable to convert")
        return None, None
    try:
        numerator, denominator = map(int, match.groups())
    except Exception:
        return None, None
    return numerator, denominator
