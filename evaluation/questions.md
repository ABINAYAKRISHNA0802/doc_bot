# RAG Evaluation Test Cases
**Source URL:** https://docs.python.org/3/tutorial/introduction.html

---

## Category 1: Direct Questions (page-ல் clearly இருக்கும்) — 5 questions

| # | Question | Expected Answer (keywords) | Category |
|---|----------|---------------------------|----------|
| 1 | What operator is used to calculate powers in Python? | `**` operator | direct |
| 2 | What is the type of a number with a fractional part in Python? | float | direct |
| 3 | How do you create a multi-line string in Python? | triple quotes (`"""` or `'''`) | direct |
| 4 | What does the // operator do in Python? | floor division, discards fractional part | direct |
| 5 | What suffix is used to indicate the imaginary part of a complex number in Python? | j or J suffix | direct |

---

## Category 2: Detailed / Specific Fact Questions — 3 questions

| # | Question | Expected Answer (keywords) | Category |
|---|----------|---------------------------|----------|
| 6 | What is the result of 17 divided by 3 using classic division in Python? | 5.666666666666667 | detailed |
| 7 | What happens when you use a variable that has not been defined in Python? | error / NameError / not defined | detailed |
| 8 | What types of numbers does Python support besides int and float? | Decimal, Fraction, complex numbers | detailed |

---

## Category 3: "Illatha" Questions (page-ல் இல்லாத info) — 3 questions

| # | Question | Expected Answer | Category |
|---|----------|-----------------|----------|
| 9  | Who created the Python programming language? | இல்லை (not on this page) | not_in_page |
| 10 | What is the latest version of Python available for download? | இல்லை (not on this page) | not_in_page |
| 11 | How do you connect Python to a database? | இல்லை (not on this page) | not_in_page |

---

## Category 4: Follow-up Questions — 2 pairs

| # | Question | Depends On | Expected Answer (keywords) | Category |
|---|----------|------------|---------------------------|----------|
| 12 | What is a string in Python? | — | sequence of characters, str type, text | followup_base |
| 13 | How do you concatenate them? | Q12 (strings) | `+` operator | followup |
| 14 | What is a list in Python? | — | ordered collection, square brackets | followup_base |
| 15 | How do you access its elements? | Q14 (list) | indexing, `[0]`, slice | followup |
