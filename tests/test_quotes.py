from leat.agent.quotes import found, quoted


def test_quoted():
    # a reply's quotes, of four words at least, each with the sources cited right after it, in
    # its sentence: [1], [2][3], or after its period, or as gpt-oss cites; not a term quoted, nor
    # one cited by none, nor by the next sentence
    reply = (
        "Die Frist: „Die Kündigungsfrist beträgt drei Monate zum Quartalsende.“ [1] Außerdem gilt "
        '"The client shall pay each invoice within thirty days" [2][3]. Ein Begriff wie '
        "„Kündigung“ [1] zählt nicht. “Either party may terminate this agreement”. [4] "
        "Und «nicht zitiert, aber lang genug». Next one [5]. "
        "»A quote as gpt-oss cites it«【6†L1-L3】, and “one cited by two sources here” [7], [8]."
    )
    assert quoted(reply) == [
        ("Die Kündigungsfrist beträgt drei Monate zum Quartalsende.", [1]),
        ("The client shall pay each invoice within thirty days", [2, 3]),
        ("Either party may terminate this agreement", [4]),
        ("A quote as gpt-oss cites it", [6]),
        ("one cited by two sources here", [7, 8]),
    ]


def test_found():
    # a quote found word for word, but for case, spacing, a word broken over lines, quotation
    # marks and dashes, and an ellipsis's parts in order; not one changed
    source = (
        "§ 3 Kündigung\n\nDie Kündigungs-\nfrist beträgt  drei Monate zum Quartalsende – "
        "schriftlich. Der Mieter nennt es „fair“."
    )
    assert found("Die Kündigungsfrist beträgt drei Monate zum Quartalsende.", source)
    assert found("die kündigungsfrist beträgt drei monate zum quartalsende - schriftlich", source)
    assert found("Die Kündigungsfrist … zum Quartalsende", source)
    assert found('Der Mieter nennt es "fair".', source)
    assert not found("Die Kündigungsfrist beträgt sechs Monate", source)
    assert not found("zum Quartalsende … Die Kündigungsfrist", source)  # out of order
