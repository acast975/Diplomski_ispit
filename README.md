# Diplomski_ispit

Vrši se analiza za skupove podataka:
- Thyroid Gland Data Set [1]
- Breast Cancer Wisconsin [2]
- Cardiotocography Data Set [3]
- Arrhythmia Data Set [4]

,a okviru foldera "Data" smešteni su ovi podaci. Breast Cancer Wisconsin skup podataka smešten je u okviru dva .csv fajla: BreastCancerWisconsin_DataSet.csv (sadrži sve uzorke podataka) i BreastCancerWisconsin_DataSet_NoMissVal.csv (iz koga su isključeni uzorci sa izgubljenim vrednostima).

Podaci su preuzeti sa sa sajta UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Reference podataka su:

[1] Danny Coomans, UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine,
    CA: University of California, School of Information and Computer Science, 1983

[2] Dr. WIlliam H. Wolberg, UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science, 1991

[3] Joao Bernardes, UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine,
    CA:University of California, School of Information and Computer Science, 2000

[4] Altay Guvenir, Burak Acar, Haldun Muderrisoglu, UCI Machine Learning Repository
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information
    and Computer Science, 1998


Kod analize za svaki skup podataka smešten je u okviru odgovarajuće python skripte. Samo je analiza nad Cardiotocography skupom podataka smeštena u okviru dve skripte, a to je urađeno iz razloga zato što se u ovome skupu podataka vrši analiza za dve različite response vrednosti. Dakle, kod analize podataka smešten je u sledećih pet python skripti:

- Arrhythmia.py
- BreastCancer.py
- Cardiotocography_CLASS.py
- Cardiotocography_NSP.py
- ThyroidDisease.py
