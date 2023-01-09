# -*- coding: utf-8 -*-
# @Time : 2023/1/9 10:28
# @Author : Stanley
# @EMail : gzlishouxian@corp.netease.com
# @File : detokenizer.py
# @Software: PyCharm
import re


class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """

    CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(more)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)(?=\s)",
    ]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]


class Detokenizer:
    def __init__(self):
        super(Detokenizer, self).__init__()
        _contractions = MacIntyreContractions()
        self.CONTRACTIONS3 = [
            re.compile(pattern.replace("(?#X)", r"\s"))
            for pattern in _contractions.CONTRACTIONS3
        ]
        self.CONTRACTIONS2 = [
            re.compile(pattern.replace("(?#X)", r"\s"))
            for pattern in _contractions.CONTRACTIONS2
        ]
        # ending quotes
        self.ENDING_QUOTES = [
            (re.compile(r"([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1\2 "),
            (re.compile(r"([^' ])\s('[sS]|'[mM]|'[dD]|') "), r"\1\2 "),
            (re.compile(r"(\S)\s(\'\')"), r"\1\2"),
            (
                re.compile(r"(\'\')\s([.,:)\]>};%])"),
                r"\1\2",
            ),  # Quotes followed by no-left-padded punctuations.
            (re.compile(r"''"), '"'),
        ]
        # Handles double dashes
        self.DOUBLE_DASHES = (re.compile(r" -- "), r"--")
        # Undo padding on parentheses.
        self.PARENS_BRACKETS = [
            (re.compile(r"([\[\(\{\<])\s"), r"\g<1>"),
            (re.compile(r"\s([\]\)\}\>])"), r"\g<1>"),
            (re.compile(r"([\]\)\}\>])\s([:;,.])"), r"\1\2"),
        ]
        # punctuation
        self.PUNCTUATION = [
            (re.compile(r"([^'])\s'\s"), r"\1' "),
            (re.compile(r"\s([?!])"), r"\g<1>"),  # Strip left pad for [?!]
            # (re.compile(r'\s([?!])\s'), r'\g<1>'),
            (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r"\1\2\3"),
            # When tokenizing, [;@#$%&] are padded with whitespace regardless of
            # whether there are spaces before or after them.
            # But during detokenization, we need to distinguish between left/right
            # pad, so we split this up.
            (re.compile(r"([#$])\s"), r"\g<1>"),  # Left pad.
            (re.compile(r"\s([;%])"), r"\g<1>"),  # Right pad.
            # (re.compile(r"\s([&*])\s"), r" \g<1> "),  # Unknown pad.
            (re.compile(r"\s\.\.\.\s"), r"..."),
            # (re.compile(r"\s([:,])\s$"), r"\1"),  # .strip() takes care of it.
            (
                re.compile(r"\s([:,])"),
                r"\1",
            ),  # Just remove left padding. Punctuation in numbers won't be padded.
        ]
        # starting quotes
        self.STARTING_QUOTES = [
            (re.compile(r"([ (\[{<])\s``"), r"\1``"),
            (re.compile(r"(``)\s"), r"\1"),
            (re.compile(r"``"), r'"'),
        ]

    def tokenize(self, tokens):
        text = " ".join(tokens)
        # Reverse the contractions regexes.
        # Note: CONTRACTIONS4 are not used in tokenization.
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r"\1\2", text)
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r"\1\2", text)

        # Reverse the regexes applied for ending quotes.
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        # Undo the space padding.
        text = text.strip()

        # Reverse the padding on double dashes.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # Reverse the padding regexes applied for parenthesis/brackets.
        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for punctuations.
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for starting quotes.
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        return text.strip()

