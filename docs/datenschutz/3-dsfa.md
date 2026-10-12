# Datenschutz-Folgenabschätzung (Art. 35 DSGVO): Vorlage, für Leat ausgefüllt

> Entwurf, nicht anwaltlich geprüft. Eine Vorlage: Das Büro prüft jede Angabe, ergänzt die Stellen in
> ⟨spitzen Klammern⟩ und entscheidet selbst, ob eine Folgenabschätzung erforderlich ist. Bei
> Berufsgeheimnisträgern, die mit neuer Technik besonders schützenswerte Mandatsdaten verarbeiten,
> spricht vieles dafür, sie durchzuführen (Art. 35 Abs. 1 DSGVO; „neue Technologien“ und
> „vertrauliche Daten“ als Kriterien der Datenschutzbehörden).

**Verantwortlicher:** ⟨Name und Anschrift des Büros⟩
**Datenschutzbeauftragte(r):** ⟨Name, falls bestellt⟩
**Durchgeführt von / am:** ⟨…⟩
**Nächste Überprüfung:** ⟨etwa jährlich, und bei jedem größeren Update von Leat⟩

## 1. Beschreibung der Verarbeitung

### 1.1 Gegenstand

Die Beschäftigten des Büros nutzen Leat, einen KI-Assistenten, für ihre Arbeit mit Akten und
Dokumenten: Fragen an Akten stellen, Dokumente zusammenfassen und vergleichen, Schreiben und
Schriftsätze entwerfen, Vorlagen ausfüllen, Tabellen auswerten, im Web recherchieren (abschaltbar).
Leat läuft vollständig auf einer Box im Büro; das KI-Modell ebenso.

### 1.2 Kategorien betroffener Personen

- Mandantinnen und Mandanten, Gegner, Zeugen und andere Beteiligte der Mandate
- Beschäftigte des Büros (als Nutzende)
- ⟨weitere, etwa Geschäftspartner des Büros⟩

### 1.3 Kategorien personenbezogener Daten

- Inhalte der Akten und Dokumente, die das Büro Leat gibt: Namen, Anschriften, Vertrags-, Finanz- und
  Verfahrensdaten; gegebenenfalls besondere Kategorien nach Art. 9 DSGVO (etwa Gesundheitsdaten in
  Personenschadensachen) und Daten über Straftaten nach Art. 10 DSGVO
- Chats: Fragen der Beschäftigten und Antworten von Leat
- Nutzungsdaten: welche Person mit welchem Gerät was getan hat (Aktivitätsprotokoll, ohne Inhalte)

### 1.4 Empfänger

- Innerhalb des Büros: die Personen, mit denen ein Projekt geteilt ist; sonst nur die Person selbst.
- Außerhalb: keine für Inhalte. Websuchen gehen an Suchanbieter, wenn das Internet für Leat
  eingeschaltet ist; sie werden vor dem Versand auf Namen aus den Akten geprüft (siehe
  [1-was-die-box-verlaesst.md](1-was-die-box-verlaesst.md)).
- Der Anbieter von Leat erhält keine Inhalte.

### 1.5 Speicherort und -dauer

- Auf der Box im Büro, verschlüsselt. Keine Übermittlung in Drittländer für Inhalte.
- Chats und Dateien: bis die Person sie löscht; frühere Fassungen von Dateien 30 Tage.
- Aktivitätsprotokoll: ein Jahr.
- ⟨Aufbewahrungsfristen des Büros für Handakten, § 50 BRAO, beachten: Leat ist ein Arbeitsmittel,
  nicht die Handakte.⟩

### 1.6 Zwecke und Rechtsgrundlage

- Zweck: effizientere Bearbeitung der Mandate; Entwürfe, Recherche, Zusammenfassungen.
- Rechtsgrundlage: ⟨Art. 6 Abs. 1 lit. b DSGVO (Mandatsvertrag) und lit. f (berechtigtes Interesse an
  effizienter Arbeit); für besondere Kategorien Art. 9 Abs. 2 lit. f DSGVO (Rechtsansprüche); für
  Beschäftigtendaten § 26 BDSG beziehungsweise Art. 88 DSGVO – anwaltlich zu prüfen⟩

## 2. Notwendigkeit und Verhältnismäßigkeit

- **Datenminimierung:** Leat verarbeitet nur, was eine Person ihm gibt oder in einem Projekt ablegt.
  Es durchsucht keine Systeme des Büros von sich aus. Projekte trennen die Mandate.
- **Speicherbegrenzung:** Chats und Dateien löscht die Person selbst; das Protokoll nach einem Jahr.
- **Keine Weitergabe zum Training:** Leat trainiert kein Modell mit den Daten des Büros, und das
  Modell läuft auf der Box.
- **Alternative geprüft:** Cloud-KI-Dienste würden Mandatsdaten an Dritte übermitteln, oft in
  Drittländer; Leat vermeidet das, indem alles im Büro bleibt.

## 3. Risiken für die Rechte und Freiheiten der Betroffenen

| Risiko | Eintritt | Schwere | Maßnahmen | Restrisiko |
|---|---|---|---|---|
| Unbefugte Kenntnis von Mandatsdaten durch Diebstahl der Box | gering | hoch | Verschlüsselung, Schlüssel im TPM nur für signierte Software (TOM 1.2) | gering |
| Unbefugte Kenntnis durch Beschäftigte ohne Mandatsbezug | mittel | hoch | Eigene Chats und Dateien; Projekte nur für ausgewählte Personen (TOM 1.3) | gering |
| Offenlegung über Websuchen (Mandantenname in der Suchanfrage) | mittel | hoch | Anweisung an das Modell; Prüfung jeder Anfrage vor dem Versand; Internet abschaltbar | gering; bei abgeschaltetem Internet keines |
| Fehlerhafte oder erfundene Inhalte („Halluzinationen“), die in ein Schreiben übernommen werden | mittel | mittel | Quellenangaben zu jeder Aussage; wörtliche Prüfung jedes Zitats gegen die Quelle; Schulung; Verantwortung bleibt beim Menschen | mittel, durch Kontrolle der Ergebnisse zu senken |
| Unbefugter Zugriff aus dem Netz | gering | hoch | Nur im Büronetz erreichbar; HTTPS; jedes Gerät einzeln zugelassen; kein Fernzugang (TOM 1.2) | gering |
| Zugriff des Anbieters | gering | hoch | Kein Fernzugang, keine Inhalte beim Anbieter, signierte und quelloffene Software | gering |
| Datenverlust (Defekt der Box) | gering | mittel | ⟨Sicherung durch das Büro, TOM 3⟩ | ⟨…⟩ |
| Überwachung der Beschäftigten über das Aktivitätsprotokoll | gering | mittel | Protokoll ohne Inhalte, nur für Admins, ein Jahr; ⟨Information der Beschäftigten, ggf. Betriebsvereinbarung⟩ | gering |

## 4. Abhilfemaßnahmen

Siehe [2-tom.md](2-tom.md). Zusätzlich durch das Büro: ⟨Schulung, Regeln zur Nutzung, Benennung der
Admins, Sicherung⟩.

## 5. Ergebnis

⟨Das Büro bewertet: Nach Umsetzung der Maßnahmen ist das Restrisiko für die Rechte und Freiheiten der
Betroffenen gering / vertretbar. Eine vorherige Konsultation der Aufsichtsbehörde (Art. 36 DSGVO) ist
nicht / ist erforderlich.⟩

**Stellungnahme der/des Datenschutzbeauftragten:** ⟨…⟩

**Unterschrift:** ⟨…⟩
