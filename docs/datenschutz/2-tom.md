# Technische und organisatorische Maßnahmen (Art. 32 DSGVO)

> Entwurf, nicht anwaltlich geprüft. Beschreibt die Maßnahmen der Leat-Box selbst. Die
> organisatorischen Maßnahmen des Büros (wer Zugang zum Raum hat, wer Admin ist, Schulung) ergänzt
> das Büro in Abschnitt 9.

**Verarbeitung:** Ein KI-Assistent für die Arbeit mit den Akten und Dateien des Büros, der vollständig
auf einer Box im Büro läuft (Leat).
**Verantwortlicher:** ⟨Name und Anschrift des Büros⟩
**Stand:** Leat-Version 2026.10

## 1. Vertraulichkeit

### 1.1 Zutrittskontrolle

- Die Box steht in den Räumen des Büros. Es gibt kein Rechenzentrum und keine Cloud, in der die Daten
  liegen.
- Die Box hat keine Anmeldung am Bildschirm und keine Tastatur-Konsole; ihr Bildschirm zeigt nur
  ihren Zustand.

### 1.2 Zugangskontrolle

- **Verschlüsselung der Datenträger:** Alle Daten des Büros liegen auf einer verschlüsselten
  Partition (LUKS2, AES-256 im XTS-Modus). Ihr Schlüssel ist im TPM-Sicherheitschip der Box versiegelt
  und wird nur freigegeben, wenn die Box die unveränderte, vom Anbieter signierte Software startet
  (signierte PCR-Richtlinie). Eine ausgebaute Festplatte oder eine mit anderer Software gestartete
  Box gibt nichts preis.
- **Geräte statt Passwörter:** Jedes Gerät, das Leat nutzt, wird einzeln zugelassen. Ein neues Gerät
  fragt den Beitritt an und zeigt einen sechsstelligen Code; eine bereits berechtigte Person oder ein
  Admin lässt es herein, nachdem sie den Code verglichen hat. Das Gerät erhält ein Geheimnis in einem
  Cookie (HttpOnly, Secure), von dem die Box nur einen Hash speichert.
- **Netzlaufwerk:** Wer seine Dateien und Projekte im Explorer oder Finder öffnen will, erzeugt in
  den Einstellungen ein Passwort für seinen Computer. Es wird einmal angezeigt, die Box speichert nur
  einen Hash, es steht bei den Geräten der Person und lässt sich dort entkoppeln. Es öffnet nur das
  Netzlaufwerk, nicht die App, und nur über HTTPS. Das Netzlaufwerk zeigt nur die eigenen Dateien und
  die Projekte, die die Person sieht.
- **Einrichtung nur vor Ort:** Die erste Einrichtung verlangt einen Code, den nur der Bildschirm der
  Box zeigt. Wer die Box nicht sieht, kann sie nicht einrichten.
- **Wiederherstellung nur vor Ort:** Haben alle Admins ihre Geräte verloren, macht ein Druck auf den
  Einschaltknopf der Box einen Wiederherstellungscode für fünf Minuten sichtbar. Die Admins sehen
  einen Hinweis in der App und können den Code widerrufen.
- **Verschlüsselte Verbindungen:** Die App ist nur über HTTPS mit einem öffentlich vertrauenswürdigen
  Zertifikat erreichbar (HSTS); Aufrufe über HTTP werden auf HTTPS umgeleitet.
- **Nur aus dem Büronetz:** Die Box ist unter einer privaten Adresse im Büronetz erreichbar, nicht
  aus dem Internet. Anfragen von fremden Webseiten (DNS-Rebinding, CSRF) werden abgewiesen.
- **Kein Fernzugang:** Die ausgelieferte Box hat keinen SSH-Zugang und kein Wartungskonto. Der
  Anbieter kann sich nicht anmelden.

### 1.3 Zugriffskontrolle

- **Eigene Chats und Dateien:** Jede Person sieht nur ihre eigenen Chats und Dateien. Auch Admins
  sehen die Chats und Dateien anderer nicht.
- **Projekte:** Ein Projekt (Mandat, Akte) gehört der Person, die es angelegt hat, und ist nur für sie
  sichtbar, bis sie es mit ausgewählten Personen oder allen teilt. Wer ein Projekt nicht sieht, kann
  auch seine Dateien nicht öffnen, durchsuchen oder von Leat lesen lassen; die Suche und die
  Auswertungen von Leat bleiben im jeweiligen Projekt.
- **Rollen:** Admins lassen Personen und Geräte herein, entfernen sie und verwalten die Einstellungen
  der Box. Mitglieder nutzen Leat. Es gibt immer mindestens einen Admin.
- **Abgeschotteter Code:** Programmcode, den Leat zur Auswertung von Dateien ausführt, läuft in einer
  Sandbox (bubblewrap) ohne Netzwerk und nur mit Zugriff auf den Arbeitsbereich des jeweiligen Chats.

### 1.4 Trennungskontrolle

- Die Daten jeder Person und jedes Projekts liegen in eigenen Bereichen; Suchindex und Auswertungen
  sind je Bereich getrennt.
- Die Box verarbeitet nur die Daten des Büros, in dem sie steht. Es gibt keine gemeinsame Plattform
  mit anderen Kunden.

### 1.5 Pseudonymisierung

- Nicht vorgesehen; die Daten verlassen die Box nicht. Websuchen werden vor dem Versand auf Namen aus
  den Akten, Namen der Personen des Büros, E-Mail-Adressen und IBANs geprüft und dann nicht gesendet
  (siehe [1-was-die-box-verlaesst.md](1-was-die-box-verlaesst.md)).

## 2. Integrität

### 2.1 Weitergabekontrolle

- Die Dateien, Chats und Anfragen an das KI-Modell verlassen die Box nicht. Das Modell läuft auf der
  Box.
- Was die Box verlässt (Websuchen, Updates, ihre private Adresse, ihr Zertifikat, die Uhrzeit), ist in
  [1-was-die-box-verlaesst.md](1-was-die-box-verlaesst.md) abschließend beschrieben. Websuchen und das
  Lesen von Webseiten lassen sich abschalten.
- Keine Telemetrie. Eine Diagnose-Datei für den Support lädt nur ein Admin selbst herunter; sie
  enthält keine Dateien, Chats oder Namen von Personen.

### 2.2 Eingabekontrolle

- **Aktivitätsprotokoll:** Die Box protokolliert, wer wann was getan hat und wo: einen Chat begonnen,
  eine Datei hochgeladen, gelöscht oder von Leat gelesen, ein Projekt geteilt, eine Person
  hereingelassen, eine Einstellung geändert. Es enthält nie, was jemand gefragt oder Leat geantwortet
  hat. Admins sehen es in den Einstellungen. Es wird ein Jahr aufbewahrt und dann gelöscht.
- **Frühere Versionen:** Wird eine Datei geändert oder gelöscht, bleibt ihre vorige Fassung 30 Tage
  erhalten und lässt sich wiederherstellen.
- **Zitate werden geprüft:** Zitiert Leat aus einer Quelle, prüft die Box das Zitat wörtlich gegen die
  zitierte Datei oder Seite und markiert es als gefunden oder nicht gefunden.

## 3. Verfügbarkeit und Belastbarkeit

- **Updates mit Rückfall:** Die Box installiert Updates in eine zweite Systempartition und startet
  nachts damit. Startet das neue System nicht einwandfrei, kehrt sie nach drei Versuchen von selbst
  zum vorigen zurück. Updates sind signiert; die Box nimmt nur vom Anbieter signierte an.
- **Selbstüberwachung:** Die Box meldet auf ihrem Bildschirm und den Admins in der App, wenn etwas
  Aufmerksamkeit braucht: fast volle Festplatte, KI-Modell nicht erreichbar, kein Internet,
  ablaufendes Zertifikat, Name im Büronetz nicht auffindbar, fehlgeschlagenes Update.
- **Datensicherung:** Leat erstellt derzeit **keine eigene Sicherung** außerhalb der Box. Die Box ist
  für die Dateien, die Leat erstellt, und für die Chats die einzige Ablage. Das Büro sichert, was es
  aufbewahren muss, in seiner eigenen Ablage (etwa indem es von Leat erstellte Dokumente dort
  speichert) oder gleicht Projekte mit einem Ordner seines Netzlaufwerks ab, sodass die Akten selbst
  dort liegen. ⟨Maßnahme des Büros eintragen.⟩
- **Stromausfall:** Die Datenbank der Chats (SQLite mit Write-Ahead-Log) und die Einstellungen der Box
  übersteht einen Stromausfall; eine Datei, die gerade hochgeladen wurde, muss gegebenenfalls erneut
  hochgeladen werden. Nach dem Wiedereinschalten startet Leat von selbst.

## 4. Löschung

- **Chats und Dateien** löscht jede Person selbst; frühere Versionen von Dateien werden nach 30 Tagen
  gelöscht.
- **Eine Person entfernen** löscht ihre Chats, ihre Geräte, ihre eigenen Dateien und Projekte. Geteilte
  Projekte gehen an eine Person über, die sie schon sieht.
- **Alles löschen** (Werkseinstellung) löscht die verschlüsselte Datenpartition und legt sie mit
  einem neuen Schlüssel neu an. Die alten Daten sind damit nicht mehr lesbar (kryptografische
  Löschung). So muss eine Box weitergegeben oder entsorgt werden.

## 5. Verfahren zur regelmäßigen Überprüfung

- Der Quellcode von Leat ist offen (MIT-Lizenz) und kann von jedem geprüft werden.
- Jede ausgelieferte Version ist aus einem bestimmten Stand des Quellcodes gebaut und signiert; die
  Version steht in den Einstellungen und auf dem Bildschirm der Box.

## 6. Auftragsverarbeitung

- Der Anbieter verarbeitet keine personenbezogenen Daten des Büros im Auftrag: Er hat keinen Zugang
  zur Box und erhält keine Inhalte. ⟨Ob dennoch ein Vertrag nötig ist, etwa für die Diagnose-Datei
  oder die Adressmeldung, prüft die Anwältin oder der Anwalt.⟩
- Richtet ein IT-Dienstleister die Box ein oder betreut sie, kann er nach Absprache Admin sein. Dann
  gelten für ihn die Regeln für Dienstleister (siehe
  [4-berufsrecht-und-ki-verordnung.md](4-berufsrecht-und-ki-verordnung.md)).

## 7. Unterauftragnehmer des Anbieters

| Dienst | Wofür | Daten |
|---|---|---|
| Hetzner Online GmbH, Deutschland | Server des Anbieters (Updates, Adressmeldung, Zertifikat) | Kennung der Box, ihre private Adresse, die öffentliche IP-Adresse des Büros |
| Let's Encrypt (Internet Security Research Group, USA) | Zertifikat der Box, von der Box direkt angefragt | Der Name der Box (öffentlich), die öffentliche IP-Adresse des Büros |

Die Box fragt außerdem öffentliche DNS-Server (Cloudflare, Quad9, Google) nach ihrem eigenen Namen,
wenn sie ihr Zertifikat erneuert; diese erfahren den Namen der Box und die öffentliche IP-Adresse des
Büros. ⟨Ob Let's Encrypt und diese DNS-Server als Unterauftragnehmer gelten, obwohl die Box sie
direkt anfragt, prüft die Anwältin oder der Anwalt.⟩

## 8. Was Leat nicht tut

- Leat gibt keine Rechtsberatung und trifft keine Entscheidungen; es entwirft, sucht und fasst
  zusammen. Die Verantwortung für jedes Ergebnis bleibt bei der Person, die es verwendet.
- Leat trainiert kein Modell mit den Daten des Büros.

## 9. Organisatorische Maßnahmen des Büros

⟨Vom Büro auszufüllen, etwa:⟩

- Wer ist Admin? ⟨Namen⟩
- Wer hat Zugang zu dem Raum, in dem die Box steht? ⟨…⟩
- Wann ist das Internet für Leat abgeschaltet? ⟨etwa: immer / nie / für bestimmte Mandate⟩
- Wie werden die Beschäftigten im Umgang mit Leat geschult (KI-Kompetenz, Art. 4 KI-VO)? ⟨…⟩
- Wie werden Geräte entfernt, wenn jemand das Büro verlässt? ⟨Admin entfernt die Person am letzten
  Arbeitstag.⟩
