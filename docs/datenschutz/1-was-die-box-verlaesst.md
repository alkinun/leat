# Was die Box verlässt – und was nie

> Entwurf, nicht anwaltlich geprüft.

Leat ist ein KI-Assistent, der vollständig auf einer kleinen Box im Büro läuft. Das KI-Modell, die
Dateien, die Chats und alles, was Leat daraus liest, liegen auf dieser Box. Es gibt keine Cloud, an
die Leat Anfragen schickt, und der Anbieter von Leat hat keinen Zugang zur Box.

## Was die Box nie verlässt

- **Die Dateien des Büros**, ob hochgeladen, über das Netzlaufwerk der Box abgelegt oder von Leat
  erstellt.
- **Die Chats**: was jemand fragt und was Leat antwortet.
- **Was Leat aus den Dateien liest**: der Suchindex, die Auszüge, die Antworten auf Fragen an viele
  Dateien.
- **Die Anfragen an das KI-Modell.** Das Modell läuft auf der Box selbst.
- **Code, den Leat ausführt**, etwa um eine Tabelle auszuwerten. Er läuft in einer abgeschotteten
  Umgebung (Sandbox) ohne Internet.
- **Wer was getan hat** (das Aktivitätsprotokoll). Es bleibt ein Jahr auf der Box und enthält nie, was
  jemand gefragt oder Leat geantwortet hat.

Es gibt keine Telemetrie und keine Nutzungsstatistik, die an den Anbieter geht.

## Was die Box verlässt

| Was | Wohin | Wann | Abschaltbar |
|---|---|---|---|
| **Websuchen**: der Suchbegriff | Suchanbieter wie Google, Bing, DuckDuckGo, Brave und Wikipedia, über eine Suchmaschine (SearXNG) auf der Box | Nur wenn Leat für eine Antwort im Web sucht | Ja: Einstellungen → Internet → Aus |
| **Aufgerufene Webseiten**: ihre Adresse | Die jeweilige Webseite | Nur wenn Leat eine Seite liest | Ja, mit der Websuche |
| **Updates**: eine Abfrage, ob es ein neues Update gibt, und sein Download | leatbox.com, der Server des Anbieters | Alle paar Stunden | Nein; die Box holt Updates nur ab, sie sendet nichts dabei |
| **Die Adresse der Box im Büronetz**: ihre private IPv4-Adresse (etwa 192.168.1.20) und ihre lokale IPv6-Adresse, mit der Kennung der Box | api.leatbox.net, der Server des Anbieters (Hetzner, Deutschland) | Wenn sie sich ändert, sonst einmal am Tag | Nein |
| **Das Zertifikat der Box** für HTTPS: ihr Name | Let's Encrypt (Internet Security Research Group, USA), direkt; den Nachweis, dass der Name ihr gehört, hinterlegt die Box beim Server des Anbieters | Ein neues Zertifikat etwa alle zwei Monate; dazwischen fragt die Box gelegentlich, ob es früher zu erneuern ist | Nein |
| **Prüfung dieses Nachweises**: DNS-Abfragen nach dem Namen der Box | Öffentliche DNS-Server von Cloudflare (1.1.1.1), Quad9 (9.9.9.9) und Google (8.8.8.8) | Mit jedem Zertifikat | Nein |
| **Ob die Box das Internet erreicht**: ein Verbindungsaufbau ohne Inhalt | api.leatbox.net, der Server des Anbieters | Jede Minute | Nein |
| **Ob die Box unter ihrem Namen gefunden wird**: DNS-Abfragen nach ihrem Namen (und, wird er nicht gefunden, nach leatbox.net) | Der DNS-Server, den das Büronetz der Box gibt (meist der Router) | Jede Minute | Nein |
| **Die Uhrzeit** | Öffentliche Zeitserver (NTP) | Regelmäßig | Nein |

**Zu den Websuchen:** Leat wird angewiesen, nach der allgemeinen Frage zu suchen, nie nach Namen oder
Einzelheiten aus den Akten. Zusätzlich prüft die Box jede Suchanfrage, bevor sie hinausgeht: Enthält
sie einen Namen aus den Dateien des Chats oder aus dem, was die Person geschrieben hat (etwa nach
„Herr“, „Frau“, „Mandantin“ oder vor „GmbH“), die Namen der Personen des Büros, den Namen des Büros, eine
E-Mail-Adresse oder eine IBAN, wird sie nicht gesendet. Ebenso prüft sie die Adresse jeder Webseite,
die Leat selbst zusammensetzt (etwa die einer Suchmaschine mit einem Namen darin); Adressen, die die
Person selbst eingefügt oder eine Suche gefunden hat, liest Leat. Leat erfährt, warum, und sucht
allgemeiner; die App zeigt an, welcher Name zurückgehalten wurde. Diese Prüfung ist ein Schutznetz, keine Garantie:
Wer ganz sicher gehen will, schaltet das Internet für Leat aus.

**Zum Namen der Box:** Die Box ist im Büro unter einer Adresse wie `https://k7x2qm.leatbox.net`
erreichbar. Dieser Name und die private Adresse der Box sind öffentlich (jedes Zertifikat wird in
öffentlichen Protokollen eingetragen). Sie verraten nichts über das Büro oder seine Arbeit: Die
private Adresse ist nur im Büronetz erreichbar, und der Name sagt nicht, wem die Box gehört.

**„Internet aus“ betrifft Leat, nicht die Box:** Ist das Internet in den Einstellungen ausgeschaltet,
sucht und liest Leat nichts im Web. Die Box selbst holt weiter Updates, meldet ihre Adresse, erneuert
ihr Zertifikat und prüft, ob sie erreichbar ist, wie oben beschrieben. Keine dieser Verbindungen
enthält Daten des Büros.

## Was der Anbieter sieht

Der Anbieter von Leat sieht nur, was sein Server ohnehin erhält: die Kennung der Box, ihre private
Adresse im Büronetz, die öffentliche IP-Adresse des Büros, von der sie sich meldet, Updates abholt
und jede Minute prüft, ob sie das Internet erreicht, und wann sie das tut. Let's Encrypt sieht den
Namen der Box und die öffentliche IP-Adresse des Büros, wenn die Box ihr Zertifikat erneuert. Er hat keinen Zugang zur Box: Es gibt kein Fernwartungskonto, keinen
Schlüssel und keine Hintertür. Hilfe bei Problemen gibt es nur über eine Diagnose-Datei, die ein Admin
des Büros selbst herunterlädt und weitergibt; sie enthält nur Angaben zum System der Box, keine
Dateien, Chats oder Namen von Personen.

## Wenn die Box gestohlen wird

Die Daten liegen verschlüsselt auf der Box. Der Schlüssel ist im Sicherheitschip der Box (TPM)
versiegelt und wird nur für die unveränderte, vom Anbieter signierte Software freigegeben. Wer die
Festplatte ausbaut oder die Box mit anderer Software startet, kommt nicht an die Daten.
