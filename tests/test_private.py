from leat.agent.private import asked, held, names


def test_names():
    # the names a text gives, as offices write them: after a title or a role, before a company's
    # form; not a noun German capitalizes, which an article goes before elsewhere
    text = (
        "Sehr geehrte Frau Dr. Anna Weber,\n\nunser Mandant Herr Hofmann hat die Kündigung der "
        "Hofmann Textil GmbH erhalten.\n\n# Kündigung Müller GmbH\n\nDie Müller & Co. KG zahlt "
        "4.500 € bis zum 1. März. Mr. John Smith of Acme Ltd agreed with the client."
    )
    assert names(text) == {"anna", "weber", "hofmann", "textil", "müller", "john", "smith", "acme"}
    assert names("Kündigung Müller GmbH") == {"kündigung", "müller"}  # a heading, alone
    assert names("Die Weißbier GmbH zahlt.") == {"weissbier"}  # as the index folds it


def test_held():
    # of a query, what must not leave the box: its words of the names, as it writes them, the
    # whole names of the box's people and office, and its e-mail addresses and IBANs
    named = {"hofmann", "weber", "yılmaz"}
    office = ["Kanzlei Weber", "Alkın Ünlü"]
    assert held("Hofmann Textil Kündigung Weber", named, office) == ["Hofmann", "Weber"]
    assert held("Kündigungsfrist Probezeit § 622 BGB", named, office) == []
    assert held("Bewertungen Kanzlei Weber", set(), office) == ["Kanzlei Weber"]
    assert held("Kanzlei Webers Bewertungen", set(), office) == []
    assert held("mail anna@weber.de, IBAN DE89 3704 0044 0532 0130 00", set(), []) == [
        "anna@weber.de", "DE89 3704 0044 0532 0130 00"]  # fmt: skip
    assert asked("Kündigung der Hofmann Textil 2024 an") == ["kündigung", "hofmann", "textil"]
