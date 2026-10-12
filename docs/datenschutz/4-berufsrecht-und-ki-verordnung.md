# Berufsrecht und KI-Verordnung: Hinweise für Kanzleien

> Entwurf, nicht anwaltlich geprüft, und keine Rechtsberatung. Diese Hinweise ordnen ein, wie Leat zu
> den Pflichten von Berufsgeheimnisträgern und zur KI-Verordnung steht. Die rechtliche Bewertung trifft
> das Büro.

## 1. Verschwiegenheit: § 203 StGB und § 43a Abs. 2 BRAO

Rechtsanwältinnen und Rechtsanwälte machen sich nach § 203 Abs. 1 Nr. 3 StGB strafbar, wenn sie ein
fremdes Geheimnis unbefugt offenbaren; § 43a Abs. 2 BRAO und § 2 BORA verpflichten sie berufsrechtlich
zur Verschwiegenheit. Für Steuerberater, Wirtschaftsprüfer und Notare gelten entsprechende Pflichten.

**Wie Leat dazu steht:** Wer eine Akte in Leat hochlädt oder Leat eine Frage zu einem Mandat stellt,
offenbart dabei niemandem etwas. Die Akte, die Frage und die Antwort bleiben auf der Box im Büro; das
KI-Modell läuft dort. Anders als bei Cloud-KI-Diensten erhält kein Dritter die Inhalte, auch nicht
der Anbieter von Leat.

**Wo dennoch etwas das Büro verlässt:**

- **Websuchen.** Sucht Leat im Web, geht der Suchbegriff an Suchanbieter. Enthielte er ein
  Mandatsgeheimnis, wäre das ein Offenbaren. Leat sucht deshalb nach der allgemeinen Rechtsfrage, und
  die Box prüft jede Suchanfrage vor dem Versand auf Namen aus den Akten, Namen der Personen des Büros,
  E-Mail-Adressen und IBANs (siehe [1-was-die-box-verlaesst.md](1-was-die-box-verlaesst.md)). Diese
  Prüfung erkennt Namen, wie Kanzleien sie schreiben („Herr Hofmann“, „Mandantin Yılmaz“, „Hofmann
  Textil GmbH“), aber nicht jede Einzelheit eines Sachverhalts. **Empfehlung:** Für besonders
  sensible Mandate, oder grundsätzlich, schaltet ein Admin das Internet für Leat in den Einstellungen
  aus. Leat arbeitet dann nur mit seinem Wissen und den Dateien des Büros.
- **Updates, Adressmeldung, Zertifikat.** Diese Verbindungen enthalten keine Mandatsdaten.

## 2. Dienstleister: § 43e BRAO und § 203 Abs. 3 und 4 StGB

Seit 2017 dürfen Berufsgeheimnisträger Dienstleistern den Zugang zu geschützten Tatsachen eröffnen,
soweit das für die Dienstleistung erforderlich ist (§ 203 Abs. 3 Satz 2 StGB, § 43e BRAO). Dafür
verlangt § 43e BRAO unter anderem eine sorgfältige Auswahl und einen Vertrag in Textform, der den
Dienstleister zur Verschwiegenheit verpflichtet und über die strafrechtlichen Folgen belehrt; für
Dienstleistungen im Ausland gilt § 43e Abs. 4 BRAO. Wer den Dienstleister nicht zur Geheimhaltung
verpflichtet, kann sich nach § 203 Abs. 4 Satz 2 Nr. 1 StGB strafbar machen.

**Der Anbieter von Leat** hat keinen Zugang zur Box und zu ihren Inhalten: kein Fernwartungskonto,
keine Inhalte auf seinen Servern, keine Telemetrie. Nach unserer Einschätzung eröffnet das Büro ihm
damit keinen Zugang zu geschützten Tatsachen. ⟨Ob dennoch eine Verpflichtung nach § 43e BRAO
vorsorglich vereinbart werden soll, etwa mit Blick auf Software-Updates oder die Diagnose-Datei,
prüft die Anwältin oder der Anwalt.⟩ Dafür spricht, dass Updates die Software der Box verändern;
dagegen, dass sie signiert sind, aus offenem Quellcode gebaut werden und keinen Zugang eröffnen.

**Der IT-Dienstleister des Büros**, der die Box aufstellt und betreut, kann Admin der Box sein. Ein
Admin sieht nicht die Chats und Dateien anderer, kann aber Personen und Geräte hereinlassen und damit
mittelbar Zugang verschaffen. Er ist daher Dienstleister im Sinne von § 43e BRAO; das Büro schließt
mit ihm den dort verlangten Vertrag, wie für seine übrige IT-Betreuung.

## 3. KI-Verordnung (Verordnung (EU) 2024/1689)

### 3.1 KI-Kompetenz (Art. 4)

Seit dem 2. Februar 2025 müssen Betreiber von KI-Systemen Maßnahmen ergreifen, um nach besten Kräften
sicherzustellen, dass ihr Personal über ausreichende KI-Kompetenz verfügt. Ein Büro, das Leat
einsetzt, ist Betreiber. Eine kurze Schulung genügt in der Regel, wenn sie dokumentiert wird.
Vorschlag für ihren Inhalt:

1. **Was Leat ist:** ein Sprachmodell, das auf der Box im Büro läuft; es formuliert wahrscheinliche
   Antworten, es „weiß“ nichts verlässlich.
2. **Grenzen:** Leat kann Fakten, Fundstellen und Zitate erfinden. Jede Aussage, die in ein Schreiben
   oder einen Schriftsatz geht, wird an der Quelle geprüft. Leat zeigt zu Aussagen seine Quellen und
   markiert Zitate, die es in der zitierten Quelle nicht wörtlich gefunden hat.
3. **Keine Rechtsberatung durch Leat:** Leat entwirft, sucht und fasst zusammen. Die rechtliche
   Bewertung und die Verantwortung bleiben bei der Anwältin oder dem Anwalt.
4. **Vertraulichkeit:** Keine Mandantendaten in Websuchen; im Zweifel das Internet für Leat
   ausschalten lassen. Akten in Projekte legen, die nur die Beteiligten sehen.
5. **Aktualität:** Das Wissen des Modells hat einen Stichtag; neue Gesetze und Urteile kennt es nur
   über eine Websuche oder die Dateien des Büros.

⟨Dokumentation: Datum, Teilnehmende, Inhalt der Schulung.⟩

### 3.2 Einstufung

Nach unserer Einschätzung ist Leat, wie Kanzleien es nutzen (Entwürfe, Recherche, Zusammenfassungen),
kein Hochrisiko-KI-System im Sinne von Art. 6 in Verbindung mit Anhang III: Anhang III Nr. 8
Buchstabe a erfasst KI-Systeme, die von Justizbehörden oder in ihrem Auftrag bei der Ermittlung und
Auslegung von Sachverhalten und Rechtsvorschriften verwendet werden sollen, nicht die Arbeit von
Kanzleien. ⟨Das Büro prüft diese Einstufung für seine eigene Nutzung.⟩

### 3.3 Transparenz (Art. 50)

Die Transparenzpflichten für Betreiber betreffen vor allem Deepfakes und KI-erzeugte Texte, die zur
Information der Öffentlichkeit über Angelegenheiten von öffentlichem Interesse veröffentlicht werden.
Für Schreiben an Mandanten, die eine Anwältin oder ein Anwalt geprüft und verantwortet, sind sie nach
unserer Einschätzung in der Regel nicht einschlägig. ⟨Anwaltlich zu prüfen.⟩

## 4. Datenschutz

Siehe [2-tom.md](2-tom.md) für die technischen und organisatorischen Maßnahmen und
[3-dsfa.md](3-dsfa.md) für die Vorlage einer Datenschutz-Folgenabschätzung.
