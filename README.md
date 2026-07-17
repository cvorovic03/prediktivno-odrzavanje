# 🏭 Prediktivno održavanje

Sistem za procjenu rizika otkaza mašine koristeći **Random Forest** i **Logistic Regression**.

## 📌 O projektu

Proizvodna kompanija trenutno servisira mašine prema unaprijed definisanom rasporedu ili tek nakon što se kvar već desi. Preventivni servisi mogu biti nepotrebni, dok neočekivani kvarovi izazivaju zastoj proizvodnje.

Cilj je razviti sistem koji procjenjuje vjerovatnoću otkaza mašine na osnovu senzorskih podataka kao što su:
- temperatura vazduha;
- temperatura procesa;
- brzina rotacije;
- obrtni moment;
- habanje alata;
- tip proizvoda ili mašine.

## 🚀 Linkovi

- **Web aplikacija:** https://huggingface.co/spaces/cvorovic03/prediktivno-odrzavanje
- **GitHub repozitorijum:** https://github.com/cvorovic03/prediktivno-odrzavanje

## ✨ Funkcionalnosti

| Funkcija | Opis |
|---|---|
| 📊 Unos | Unos senzorskih vrijednosti za jednu mašinu |
| 📁 Upload CSV | Analiza više mašina odjednom |
| 📈 Rangiranje | Rangiranje mašina prema prioritetu za servis |
| 🔬 What-if simulacija | Simulacija uticaja promjene parametara |
| 📊 Poređenje modela | Logistic Regression vs Random Forest |
| 🔍 SHAP objašnjenja | Objašnjenje zašto je mašina rizična |

## 🛠️ Tehnologije

- **Python 3.9**
- **Streamlit** – web interfejs
- **Scikit-learn** – Logistic Regression, Random Forest
- **XGBoost** – mašinsko učenje
- **SHAP** – objašnjivost modela
- **Plotly** – vizualizacije
- **Pandas, NumPy** – obrada podataka

## 📊 Health Score

**Health Score od 0 do 100** – što je veći, mašina je zdravija:

| Score | Status | Akcija |
|---|---|---|
| 85 - 100 | ✅ Normalan rad | Nastaviti praćenje |
| 65 - 84 | 📊 Dodatni nadzor | Pojačati monitoring |
| 40 - 64 | 🔧 Planirati servis | U narednih 7 dana |
| 0 - 39 | 🚨 Hitna provjera | Odmah zaustaviti mašinu |

## 📁 Instalacija i pokretanje

```bash
# Kloniranje repozitorijuma
git clone https://github.com/cvorovic03/prediktivno-odrzavanje.git
cd prediktivno-odrzavanje

# Instalacija zavisnosti
pip install -r requirements.txt

# Treniranje modela
python train_model.py

# Pokretanje aplikacije
streamlit run app.py
