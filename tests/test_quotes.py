from leat.agent.quotes import found, quoted


def test_quoted():
    # a reply's quotes, of four words at least, each with the sources cited right after it, in
    # its sentence: [1], [2][3], or after its period, or as gpt-oss cites; not a term quoted, nor
    # one cited by none, nor by the next sentence
    reply = (
        "Die Frist: „Die Kündigungsfrist beträgt drei Monate zum Quartalsende.“ [1] Außerdem gilt "
        '"The client shall pay each invoice within thirty days" [2][3]. Ein Begriff wie '
        "„Kündigung“ [1] zählt nicht. “Either party may terminate this agreement”. [4] "
        "Und «nicht zitiert, zu kurz». Next one [5]. "
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


def test_quoted_blocks():
    # a block quoted, as models quote: of its own citations, of those right after it, or of none;
    # a German quote closed of a plain mark, and one in bold
    reply = (
        "Section 19 says:\n\n> Either party may terminate this agreement by giving ninety (90) "
        "days' written notice.\n\nSo, 90 days. Und: **„Die Kündigungsfrist beträgt drei Monate zum "
        'Quartalsende."** [1]\n\n> „Die Miete beträgt 900 EUR“\n> im Monat.\n\n[2]'
    )
    assert quoted(reply) == [
        (
            "Either party may terminate this agreement by giving ninety (90) days' written notice.",
            [],
        ),
        ("Die Miete beträgt 900 EUR“ im Monat.", [2]),
        ("Die Kündigungsfrist beträgt drei Monate zum Quartalsende.", [1]),
    ]
    assert found("„Die Miete beträgt 900 EUR“ im Monat.", "Die Miete beträgt 900 EUR im Monat.")
    # a sentence quoted, cited by none, as models forget to: checked against every source
    assert quoted('It says: *"Either party may terminate this agreement by giving notice."*') == [
        ("Either party may terminate this agreement by giving notice.", [])]  # fmt: skip
