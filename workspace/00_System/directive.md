# Forschungsprojekt: OrbusSim — pH-sensitive Fluoreszenzkinetik

## Zielsetzung

Das primäre Ziel dieses Forschungsprojekts ist die **Optimierung der Reagenzien-Zusammensetzung und Prozessparameter** einer Ascorbinsäure-getriebenen Fenton-Reaktion, um ein definiertes Fluoreszenz-Kinetikprofil zu erreichen.

Das System soll die optimalen Parameter finden, um:
- Eine **schnelle und reproduzierbare pH-Absenkung** von pH 7.4 auf einen Zielbereich von pH 5.0–6.0 zu erreichen.
- Ein **stabiles Fluoreszenzsignal** mit hoher Signal-zu-Rausch-Verhältnis zu erzeugen.
- Die **Reaktionskinetik** so zu steuern, dass das Fluoreszenzplateau innerhalb der Messdauer erreicht wird.

## Kritische Randbedingungen

### Thermische Degradation
Die Probentemperatur darf **75.0 °C** während des Prozesses unter keinen Umständen überschreiten, da Fluorescein bei hohen Temperaturen irreversibel degradiert.

### Material-Effizienz
Die Menge des teuren Fluorescein-Farbstoffs soll so gering wie möglich gehalten werden, idealerweise unter **5 µM Endkonzentration**, ohne das Signal-Rausch-Verhältnis unzulässig zu verschlechtern.

### Mischgüte
Die Probe muss vor der Temperaturinkubation ausreichend homogenisiert sein (mindestens **5 Sekunden bei > 300 RPM**), um Konzentrationsgradienten zu vermeiden.

### Photobleaching-Schutz
Die Anregungsleistung und Messdauer müssen so gewählt werden, dass das Photobleaching des Fluorophors während der Messung **unter 20 %** des Initialsignals bleibt.

## Datenphilosophie

**Wichtig:** Das System erzeugt bewusst nur **Rohdaten**. Es werden keine korrigierten Fluoreszenzwerte, keine berechneten pH-Werte und keine Fe²⁺-Konzentrationen als Output geliefert. Die KI muss aus den rohen Fluoreszenz-Zeitreihen und Temperaturprofilen selbst die relevanten Merkmale extrahieren und die Reaktionskinetik interpretieren.

## Experimentelle Strategie

Die KI soll iterative Experimente planen, um den optimalen Bereich zwischen folgenden Parametern zu finden:
- **Ascorbinsäure-Konzentration** (Reduktionsmittel, bestimmt Reaktionsgeschwindigkeit)
- **FeCl₃-Konzentration** (Katalysator, bestimmt Fe²⁺-Verfügbarkeit)
- **H₂O₂-Konzentration** (Co-Oxidationsmittel, bestimmt Fenton-Reaktionsrate)
- **Fluorescein-Konzentration** (Reporter-Molekül, bestimmt Signalstärke)
- **Phosphat-Puffer-Konzentration** (pH-Stabilisierung, bestimmt Pufferkapazität)
- **Temperatur** (beeinflusst Reaktionskinetik und Fluoreszenzquantenausbeute)
- **Mischgeschwindigkeit und -dauer** (beeinflusst Homogenität)